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

## Benchmarking

With the server running, execute the benchmark suite in a separate terminal:

```bash
python benchmarks/bench.py
```

This runs four workloads against the server:

| Workload | What it measures |
|---|---|
| Upsert throughput | Vectors/sec at varying concurrency levels |
| Search latency | p50/p95/p99 latency and queries/sec |
| Recall@10 | HNSW approximation accuracy vs brute-force |
| Compaction | Time to reclaim dead vectors from disk |
| Mixed 80/20 | Read/write contention over a timed window |

Run `python benchmarks/bench.py --help` for all options.
