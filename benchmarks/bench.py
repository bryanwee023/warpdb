"""WarpDB benchmark suite.

Measures upsert throughput, search latency, recall@k, and mixed workload
performance against a running WarpDB server.

Usage:
    poetry run uvicorn warpdb.api.server:app        # in one terminal
    python benchmarks/bench.py --sizes 10000 100000  # in another
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 768
DEFAULT_URL = "http://localhost:8000"
DEFAULT_SIZES = [200] # TODO: Increase for full benchmark: [10_000, 100_000]
DEFAULT_CONCURRENCY = [1] # TODO: Increase for full benchmark: [1, 8, 32, 128]
SEARCH_QUERIES = 500
RECALL_QUERIES = 100
RECALL_K = 10

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UpsertResult:
    dataset_size: int
    concurrency: int
    total_seconds: float
    vectors_per_sec: float
    errors: int


@dataclass
class SearchResult:
    dataset_size: int
    concurrency: int
    num_queries: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    errors: int


@dataclass
class RecallResult:
    dataset_size: int
    k: int
    recall: float


@dataclass
class CompactResult:
    dataset_size: int
    dead_ratio: float
    total_seconds: float


@dataclass
class MixedResult:
    dataset_size: int
    concurrency: int
    total_ops: int
    ops_per_sec: float
    read_p50_ms: float
    read_p95_ms: float
    write_p50_ms: float
    write_p95_ms: float
    errors: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_vectors(n: int, dim: int = DIM, seed: int = 42) -> np.ndarray:
    """Generate *n* unit-normalised random vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_name(i: int, prefix: str = "vec") -> str:
    return f"{prefix}-{i:08d}"


def percentile(latencies: List[float], p: float) -> float:
    if not latencies:
        return 0.0
    return float(np.percentile(latencies, p))


# ---------------------------------------------------------------------------
# Benchmark: Upsert throughput
# ---------------------------------------------------------------------------

async def bench_upsert(
    base_url: str,
    vectors: np.ndarray,
    concurrency: int,
) -> UpsertResult:
    n = len(vectors)
    sem = asyncio.Semaphore(concurrency)
    errors = 0

    async def _put(client: httpx.AsyncClient, i: int) -> None:
        nonlocal errors
        async with sem:
            try:
                resp = await client.put(
                    f"{base_url}/vectors/{make_name(i)}",
                    json={"vector": vectors[i].tolist()},
                )
                if resp.status_code != 200:
                    errors += 1
            except httpx.HTTPError:
                errors += 1

    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        start = time.perf_counter()
        await asyncio.gather(*[_put(client, i) for i in range(n)])
        elapsed = time.perf_counter() - start

    return UpsertResult(
        dataset_size=n,
        concurrency=concurrency,
        total_seconds=elapsed,
        vectors_per_sec=n / elapsed,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Benchmark: Search latency
# ---------------------------------------------------------------------------

async def bench_search(
    base_url: str,
    query_vectors: np.ndarray,
    k: int,
    concurrency: int,
) -> SearchResult:
    n = len(query_vectors)
    sem = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    errors = 0

    async def _search(client: httpx.AsyncClient, i: int) -> None:
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    f"{base_url}/search",
                    json={"vector": query_vectors[i].tolist(), "k": k},
                )
                if resp.status_code != 200:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            latencies.append(time.perf_counter() - t0)

    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        start = time.perf_counter()
        await asyncio.gather(*[_search(client, i) for i in range(n)])
        wall = time.perf_counter() - start

    return SearchResult(
        dataset_size=0,  # filled by caller
        concurrency=concurrency,
        num_queries=n,
        p50_ms=percentile(latencies, 50) * 1000,
        p95_ms=percentile(latencies, 95) * 1000,
        p99_ms=percentile(latencies, 99) * 1000,
        qps=n / wall,
        errors=errors,
    )

# ---------------------------------------------------------------------------
# Benchmark: Recall@k
# ---------------------------------------------------------------------------

async def bench_recall(
    base_url: str,
    db_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> RecallResult:
    total_recall = 0.0

    async with httpx.AsyncClient(timeout=120.0) as client:
        for qi in range(len(query_vectors)):
            q = query_vectors[qi]

            # Brute-force ground truth (squared L2)
            diffs = db_vectors - q
            dists = np.sum(diffs * diffs, axis=1)
            true_indices = np.argsort(dists)[:k]
            true_names = {make_name(int(idx)) for idx in true_indices}

            resp = await client.post(
                f"{base_url}/search",
                json={"vector": q.tolist(), "k": k},
            )
            api_names = {r["name"] for r in resp.json()}

            total_recall += len(true_names & api_names) / k

    return RecallResult(
        dataset_size=len(db_vectors),
        k=k,
        recall=total_recall / len(query_vectors),
    )


# ---------------------------------------------------------------------------
# Benchmark: Mixed workload
# ---------------------------------------------------------------------------

async def bench_mixed(
    base_url: str,
    dataset_size: int,
    concurrency: int,
    duration: float = 10.0,
    read_ratio: float = 0.8,
) -> MixedResult:
    rng = np.random.default_rng(99)
    read_lats: List[float] = []
    write_lats: List[float] = []
    errors = 0
    counter = dataset_size  # next write index (DB already has 0..dataset_size-1)
    stop = asyncio.Event()
    counter_lock = asyncio.Lock()

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal errors, counter
        while not stop.is_set():
            is_read = rng.random() < read_ratio
            if is_read:
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                t0 = time.perf_counter()
                try:
                    resp = await client.post(
                        f"{base_url}/search",
                        json={"vector": vec.tolist(), "k": 10},
                    )
                    if resp.status_code != 200:
                        errors += 1
                except httpx.HTTPError:
                    errors += 1
                read_lats.append(time.perf_counter() - t0)
            else:
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                async with counter_lock:
                    idx = counter
                    counter += 1
                t0 = time.perf_counter()
                try:
                    resp = await client.put(
                        f"{base_url}/vectors/{make_name(idx, prefix='mix')}",
                        json={"vector": vec.tolist()},
                    )
                    if resp.status_code != 200:
                        errors += 1
                except httpx.HTTPError:
                    errors += 1
                write_lats.append(time.perf_counter() - t0)

    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        workers = [asyncio.create_task(_worker(client)) for _ in range(concurrency)]
        await asyncio.sleep(duration)
        stop.set()
        await asyncio.gather(*workers)

    total_ops = len(read_lats) + len(write_lats)
    return MixedResult(
        dataset_size=dataset_size,
        concurrency=concurrency,
        total_ops=total_ops,
        ops_per_sec=total_ops / duration,
        read_p50_ms=percentile(read_lats, 50) * 1000,
        read_p95_ms=percentile(read_lats, 95) * 1000,
        write_p50_ms=percentile(write_lats, 50) * 1000,
        write_p95_ms=percentile(write_lats, 95) * 1000,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Benchmark: Compaction
# ---------------------------------------------------------------------------

async def bench_compact(
    base_url: str,
    dataset_size: int,
    dead_ratio: float = 0.25,
) -> CompactResult:
    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.perf_counter()
        resp = await client.post(f"{base_url}/compact?threshold=1.0")
        elapsed = time.perf_counter() - start
        resp.raise_for_status()

    return CompactResult(
        dataset_size=dataset_size,
        dead_ratio=dead_ratio,
        total_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

async def cleanup(base_url: str) -> None:
    async with httpx.AsyncClient(timeout=120.0) as client:
        await client.delete(f"{base_url}/vectors")


async def delete_names(base_url: str, names: List[str], concurrency: int = 8) -> None:
    """Delete specific vectors by name (used where partial deletion is needed)."""
    sem = asyncio.Semaphore(concurrency)

    async def _del(client: httpx.AsyncClient, name: str) -> None:
        async with sem:
            try:
                await client.delete(f"{base_url}/vectors/{name}")
            except httpx.HTTPError:
                pass

    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        await asyncio.gather(*[_del(client, n) for n in names])


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_upsert_table(results: List[UpsertResult]) -> None:
    print("\n=== Upsert Throughput ===")
    print(f" {'Dataset':>8} | {'Concurrency':>11} | {'Vec/sec':>9} | {'Total (s)':>9} | {'Errors':>6}")
    print(f"{'-' * 9}+{'-' * 13}+{'-' * 11}+{'-' * 11}+{'-' * 8}")
    for r in results:
        print(
            f" {r.dataset_size:>8,} | {r.concurrency:>11} | {r.vectors_per_sec:>9.1f} "
            f"| {r.total_seconds:>9.1f} | {r.errors:>6}"
        )


def print_search_table(results: List[SearchResult]) -> None:
    print("\n=== Search Latency ===")
    print(
        f" {'Dataset':>8} | {'Concurrency':>11} | {'p50 (ms)':>8} | {'p95 (ms)':>8} "
        f"| {'p99 (ms)':>8} | {'QPS':>8} | {'Errors':>6}"
    )
    print(
        f"{'-' * 9}+{'-' * 13}+{'-' * 10}+{'-' * 10}"
        f"+{'-' * 10}+{'-' * 10}+{'-' * 8}"
    )
    for r in results:
        print(
            f" {r.dataset_size:>8,} | {r.concurrency:>11} | {r.p50_ms:>8.2f} | {r.p95_ms:>8.2f} "
            f"| {r.p99_ms:>8.2f} | {r.qps:>8.1f} | {r.errors:>6}"
        )


def print_recall_table(results: List[RecallResult]) -> None:
    print("\n=== Search Recall ===")
    print(f" {'Dataset':>8} | {'Recall@' + str(results[0].k) if results else 'Recall':>10}")
    print(f"{'-' * 9}+{'-' * 12}")
    for r in results:
        print(f" {r.dataset_size:>8,} | {r.recall:>10.3f}")


def print_compact_table(results: List[CompactResult]) -> None:
    print("\n=== Compaction ===")
    print(f" {'Dataset':>8} | {'Dead %':>6} | {'Time (s)':>9}")
    print(f"{'-' * 9}+{'-' * 8}+{'-' * 11}")
    for r in results:
        print(f" {r.dataset_size:>8,} | {r.dead_ratio * 100:>5.1f}% | {r.total_seconds:>9.3f}")


def print_mixed_table(results: List[MixedResult]) -> None:
    print("\n=== Mixed Workload (80/20 read/write) ===")
    print(
        f" {'Dataset':>8} | {'Concurrency':>11} | {'Ops/sec':>8} | {'Rd p50':>8} | {'Rd p95':>8} "
        f"| {'Wr p50':>8} | {'Wr p95':>8} | {'Errors':>6}"
    )
    print(
        f"{'-' * 9}+{'-' * 13}+{'-' * 10}+{'-' * 10}+{'-' * 10}"
        f"+{'-' * 10}+{'-' * 10}+{'-' * 8}"
    )
    for r in results:
        print(
            f" {r.dataset_size:>8,} | {r.concurrency:>11} | {r.ops_per_sec:>8.1f} | {r.read_p50_ms:>8.2f} "
            f"| {r.read_p95_ms:>8.2f} | {r.write_p50_ms:>8.2f} | {r.write_p95_ms:>8.2f} | {r.errors:>6}"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_benchmarks(
    base_url: str,
    sizes: List[int],
    concurrency_levels: List[int],
    search_queries: int,
    skip_recall: bool,
    skip_mixed: bool,
    mixed_duration: float,
) -> None:
    # Health check
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{base_url}/health")
            info = resp.json()
            print(f"Server OK — {info['count']} vectors in DB")
        except Exception as e:
            print(f"Cannot reach server at {base_url}: {e}")
            return

    upsert_results: List[UpsertResult] = []
    search_results: List[SearchResult] = []
    recall_results: List[RecallResult] = []
    compact_results: List[CompactResult] = []
    mixed_results: List[MixedResult] = []

    for size in sizes:
        print(f"\n{'=' * 60}")
        print(f"  Dataset size: {size:,}")
        print(f"{'=' * 60}")

        vectors = generate_vectors(size)
        query_vectors = generate_vectors(search_queries, seed=123)

        # --- Upsert at each concurrency level ---
        for conc in concurrency_levels:
            print(f"\n  Upserting {size:,} vectors (concurrency={conc}) ...")
            result = await bench_upsert(base_url, vectors, conc)
            upsert_results.append(result)
            print(f"    {result.vectors_per_sec:.1f} vec/sec, {result.total_seconds:.1f}s, {result.errors} errors")

            # --- Search against the populated DB ---
            print(f"  Searching ({search_queries} queries, concurrency={conc}) ...")
            sr = await bench_search(base_url, query_vectors, RECALL_K, conc)
            sr.dataset_size = size
            search_results.append(sr)
            print(f"    p50={sr.p50_ms:.2f}ms  p95={sr.p95_ms:.2f}ms  p99={sr.p99_ms:.2f}ms  QPS={sr.qps:.1f}")

            # Clean up for next concurrency run
            print(f"  Cleaning up {size:,} vectors ...")
            await cleanup(base_url)

        # --- Recall ---
        if not skip_recall:
            print(f"\n  Measuring recall@{RECALL_K} ({RECALL_QUERIES} queries) ...")
            # Re-insert for recall measurement
            await bench_upsert(base_url, vectors, concurrency=max(concurrency_levels))
            recall_query_vectors = generate_vectors(RECALL_QUERIES, seed=777)
            rr = await bench_recall(base_url, vectors, recall_query_vectors, RECALL_K)
            recall_results.append(rr)
            print(f"    Recall@{RECALL_K} = {rr.recall:.3f}")
            # Clean up
            await cleanup(base_url)

        # --- Compaction ---
        dead_ratio = 0.25
        num_dead = int(size * dead_ratio)
        print(f"\n  Compaction ({num_dead} dead of {size}, {dead_ratio:.0%}) ...")
        await bench_upsert(base_url, vectors, concurrency=max(concurrency_levels))
        dead_names = [make_name(i) for i in range(num_dead)]
        await delete_names(base_url, dead_names)
        cr = await bench_compact(base_url, size, dead_ratio)
        compact_results.append(cr)
        print(f"    {cr.total_seconds:.3f}s")
        await cleanup(base_url)

        # --- Mixed workload ---
        if not skip_mixed:
            conc = max(concurrency_levels)
            print(f"\n  Mixed workload ({mixed_duration}s, concurrency={conc}) ...")
            # Populate first
            await bench_upsert(base_url, vectors, concurrency=conc)
            mr = await bench_mixed(base_url, size, conc, duration=mixed_duration)
            mixed_results.append(mr)
            print(
                f"    {mr.ops_per_sec:.1f} ops/sec  "
                f"read p50={mr.read_p50_ms:.2f}ms  write p50={mr.write_p50_ms:.2f}ms"
            )
            # Clean up all: original vectors + mixed writes
            await cleanup(base_url)

    # --- Summary tables ---
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    if upsert_results:
        print_upsert_table(upsert_results)
    if search_results:
        print_search_table(search_results)
    if recall_results:
        print_recall_table(recall_results)
    if compact_results:
        print_compact_table(compact_results)
    if mixed_results:
        print_mixed_table(mixed_results)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WarpDB Benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
        help="Dataset sizes to test (e.g. 1000 10000)",
    )
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY,
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--search-queries", type=int, default=SEARCH_QUERIES,
        help="Number of search queries for latency test",
    )
    parser.add_argument("--skip-recall", action="store_true", help="Skip recall measurement")
    parser.add_argument("--skip-mixed", action="store_true", help="Skip mixed workload test")
    parser.add_argument(
        "--mixed-duration", type=float, default=10.0,
        help="Duration of mixed workload test in seconds",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmarks(
            base_url=args.url,
            sizes=args.sizes,
            concurrency_levels=args.concurrency,
            search_queries=args.search_queries,
            skip_recall=args.skip_recall,
            skip_mixed=args.skip_mixed,
            mixed_duration=args.mixed_duration,
        )
    )


if __name__ == "__main__":
    main()
