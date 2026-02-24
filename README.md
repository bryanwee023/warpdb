# WarpDB

A vector database implemented from scratch in Python. Stores float32 vectors with optional metadata, supports approximate nearest-neighbour search via HNSW, and survives crashes using a write-ahead log.

## Architecture

```
┌──────────────────────────────────────────┐
│             FastAPI HTTP layer           │
├──────────────────────────────────────────┤
│                 WarpDB                   │
│ ┌───────────┐ ┌───────────┐ ┌──────────┐ │
│ │   HNSW    │ │   Vector  │ │ Metadata │ │
│ │   index   │ │   store   │ │  store   │ │
│ │(in-memory)│ │(.f32 file)│ │ (SQLite) │ │
│ └───────────┘ └───────────┘ └──────────┘ │
│               ┌───────────┐              │
│               │    WAL    │              │
│               │(.bin file)│              │
│               └───────────┘              │
└──────────────────────────────────────────┘
```

| Component | Implementation |
|---|---|
| Vector storage | Flat binary `.f32` file, memory-mapped for reads |
| Metadata storage | SQLite |
| ANN index | HNSW (Hierarchical Navigable Small World) |
| Crash recovery | Write-ahead log with `UPSERT`/`COMMIT` records |

## Installation

```bash
poetry install
```

## Running the server

```bash
poetry run uvicorn warpdb.api.server:app --reload
```

The server starts on `http://localhost:8000` with vector dimension fixed at 768.

## Running tests

```bash
poetry run pytest
```
