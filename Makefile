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

lake-up:          ## start the S3-compatible object store (SeaweedFS) on localhost:8333
	docker compose up -d --wait s3

run-s3:           ## full pipeline with the lake on S3 (needs lake-up)
	DE_PATHS__LAKE_URI=s3://lakehouse AWS_ENDPOINT_URL=http://localhost:8333 \
	AWS_ACCESS_KEY_ID=lakeadmin AWS_SECRET_ACCESS_KEY=lakeadmin-secret AWS_REGION=us-east-1 \
	de-pipeline run

ddl:              ## Hive Metastore / Glue DDL for all lake tables
	de-pipeline catalog --ddl
