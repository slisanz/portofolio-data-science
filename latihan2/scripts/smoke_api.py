"""Smoke test: ukur latensi /recommend terhadap server lokal."""
from __future__ import annotations

import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def _get(path: str) -> float:
    t0 = time.perf_counter()
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        r.read()
    return (time.perf_counter() - t0) * 1000


def main():
    _get("/health")
    lat = [_get("/recommend/1?k=10") for _ in range(20)]
    p50 = statistics.median(lat)
    p95 = sorted(lat)[int(0.95 * len(lat)) - 1]
    print(f"/recommend/1?k=10 p50={p50:.1f}ms p95={p95:.1f}ms n={len(lat)}")
    assert p50 < 200, f"p50 {p50:.1f}ms > 200ms"
    print("OK")


if __name__ == "__main__":
    main()
