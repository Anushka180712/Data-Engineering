.PHONY: install install-neural run search eval catalog test docker clean

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
