"""
Measure end-to-end /predict latency (TC12, NFR1).

Usage:
    python measure_latency.py --payload payload.json --label native
    python measure_latency.py --payload payload.json --label docker

Each request opens a fresh HTTP connection, like the curl call in the
GitHub Actions workflow, so the figure is the full round trip a pipeline sees.
"""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime

import requests


def percentile(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    # 127.0.0.1, not localhost: on Windows "localhost" can try IPv6 first and add delay
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--payload", required=True, help="JSON body for POST /predict")
    ap.add_argument("-n", type=int, default=500, help="measured requests")
    ap.add_argument("--warmup", type=int, default=20, help="unmeasured requests first")
    ap.add_argument("--label", default="run", help="e.g. native, docker")
    args = ap.parse_args()

    with open(args.payload) as f:
        payload = json.load(f)

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print("Health:", health)

    for _ in range(args.warmup):
        requests.post(f"{args.url}/predict", json=payload, timeout=10).raise_for_status()

    client_ms, server_ms = [], []
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = requests.post(f"{args.url}/predict", json=payload, timeout=10)
        t1 = time.perf_counter()
        r.raise_for_status()
        client_ms.append((t1 - t0) * 1000)
        server = r.json().get("latency_ms")
        if server is not None:
            server_ms.append(server)

    def summarise(vals):
        s = sorted(vals)
        return {
            "n": len(s),
            "mean": round(statistics.mean(s), 1),
            "p50": round(percentile(s, 50), 1),
            "p95": round(percentile(s, 95), 1),
            "p99": round(percentile(s, 99), 1),
            "max": round(s[-1], 1),
        }

    result = {
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "machine": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
        },
        "health": health,
        "end_to_end_ms": summarise(client_ms),
        "server_reported_ms": summarise(server_ms) if server_ms else None,
    }

    print(json.dumps(result, indent=2))
    out = f"latency_{args.label}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
