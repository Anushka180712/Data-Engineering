"""Storage abstraction for the data lake: local filesystem or S3-compatible object storage.

The same code writes to `data/lake/...` on a laptop and to `s3://bucket/...` on AWS S3 or
MinIO; only `paths.lake_uri` changes. This is the property that made HDFS -> S3 migrations
possible: compute talks to a filesystem *interface*, not to disks.

S3 settings come from the standard AWS environment variables:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (default us-east-1)
  AWS_ENDPOINT_URL   - set for MinIO / other S3-compatible stores, e.g. http://minio:9000
"""
from __future__ import annotations

import os
from pathlib import Path

import pyarrow.fs as pafs

from .config import Config


def _endpoint() -> str | None:
    return os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")


def delta_storage_options(cfg: Config) -> dict[str, str] | None:
    """Options for delta-rs (object_store). None for the local filesystem."""
    if not cfg.paths.lake_is_remote:
        return None
    opts = {"AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
            # S3 has no atomic rename/put-if-absent on older stores; we run a single writer.
            # For multiple concurrent writers on AWS use a DynamoDB lock (AWS_S3_LOCKING_PROVIDER).
            "AWS_S3_ALLOW_UNSAFE_RENAME": "true"}
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        if os.environ.get(k):
            opts[k] = os.environ[k]
    if ep := _endpoint():
        opts["AWS_ENDPOINT_URL"] = ep
        if ep.startswith("http://"):
            opts["AWS_ALLOW_HTTP"] = "true"
    return opts


def filesystem(cfg: Config) -> tuple[pafs.FileSystem, str]:
    """A pyarrow filesystem + base path for the lake root (for non-table files, e.g. raw zone)."""
    uri = cfg.paths.lake_uri
    if not cfg.paths.lake_is_remote:
        return pafs.LocalFileSystem(), uri
    if not uri.startswith("s3://"):
        raise ValueError(f"unsupported lake scheme: {uri} (use a local path or s3://)")
    fs = pafs.S3FileSystem(
        access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region=os.environ.get("AWS_REGION", "us-east-1"),
        endpoint_override=_endpoint(),
    )
    return fs, uri.removeprefix("s3://")


def lake_uri(cfg: Config, *parts: str) -> str:
    return "/".join([cfg.paths.lake_uri, *parts])


def put_file(cfg: Config, local: Path, *parts: str) -> str:
    """Copy a local file into the lake (raw zone). Returns its full lake URI."""
    fs, base = filesystem(cfg)
    dest = "/".join([base, *parts])
    fs.create_dir(dest.rsplit("/", 1)[0], recursive=True)
    with open(local, "rb") as src, fs.open_output_stream(dest) as out:
        while block := src.read(1 << 20):
            out.write(block)
    return lake_uri(cfg, *parts)


def delete_prefix(cfg: Config, *parts: str) -> None:
    fs, base = filesystem(cfg)
    try:
        fs.delete_dir("/".join([base, *parts]))
    except (FileNotFoundError, OSError):
        pass


def to_hadoop_uri(uri: str) -> str:
    """s3:// -> s3a:// (the Hadoop S3A connector scheme used by Spark/Hive); local -> file://"""
    if uri.startswith("s3://"):
        return "s3a://" + uri.removeprefix("s3://")
    return uri if "://" in uri else "file://" + uri
