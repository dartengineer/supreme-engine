# The CV generator API. Used by docker-compose.yml together with the database (pgvector) and PDF renderer.
FROM python:3.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch first: the default Linux build bundles GPU libraries and adds several GB.
# requirements.txt pins torch==2.15.0, which the +cpu build already satisfies, so pip won't replace it.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
    && pip install -r requirements.txt

# Bake the embedding model into the image, so containers start without downloading anything.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app
COPY --chown=app:app . .
USER app

EXPOSE 8000
# Bring the database schema up to date, then start the API. No secrets are baked in: settings come from
# environment variables (see docker-compose.yml).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
