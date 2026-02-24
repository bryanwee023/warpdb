# Production server image for WarpDB.
#
# Installs only production dependencies (FastAPI, Uvicorn, NumPy, Pydantic)
# and runs the ASGI server on port 8000.
#
# Usage:
#   docker build -t warpdb .
#   docker run -p 8000:8000 -v warpdb-data:/data warpdb

FROM python:3.12-slim

WORKDIR /app

# Install poetry (build tool required by pyproject.toml)
RUN pip install --no-cache-dir poetry

# Copy dependency manifests first for layer caching
COPY pyproject.toml poetry.lock ./

# Install production deps only (no dev group, no virtualenv inside container)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-root

# Copy application source
COPY src/ src/

# Install the project itself (makes "warpdb" importable)
RUN poetry install --only main --no-interaction

# Persistent storage for vectors, metadata, and WAL
VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "warpdb.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
