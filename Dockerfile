FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch keeps the image small; sentence-transformers gives neural embeddings.
ARG WITH_NEURAL=true
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ "$WITH_NEURAL" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install ".[neural]"; \
    else pip install . ; fi

COPY config ./config
VOLUME ["/app/data"]
ENTRYPOINT ["de-pipeline"]
CMD ["run"]
