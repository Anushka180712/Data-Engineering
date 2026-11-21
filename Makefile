.PHONY: install install-neural run search eval catalog test docker clean lake-up run-s3 ddl

install:          ## core install (offline LSA embeddings)
	pip install -e ".[dev]"

install-neural:   ## + sentence-transformers (neural embeddings, recommended)
	pip install -e ".[dev,neural]"

run:              ## full pipeline: ingest -> clean -> chunk -> embed -> index
	de-pipeline run

search:           ## make search Q="interest rates and inflation"
	de-pipeline search "$(Q)"

eval:
	de-pipeline eval

catalog:
	de-pipeline catalog

test:
	pytest -q

docker:
	docker compose run --rm pipeline

clean:            ## wipe all generated data
	rm -rf data/

lake-up:          ## start MinIO (S3) and create the lake bucket
	docker compose up -d minio minio-init

run-s3:           ## full pipeline with the lake on MinIO (needs lake-up)
	DE_PATHS__LAKE_URI=s3://lakehouse AWS_ENDPOINT_URL=http://localhost:9000 \
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_REGION=us-east-1 \
	de-pipeline run

ddl:              ## Hive Metastore / Glue DDL for all lake tables
	de-pipeline catalog --ddl
