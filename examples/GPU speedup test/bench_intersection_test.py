#!/usr/bin/env python3
"""
Benchmark intersection_test.test() across problem sizes with/without GPU.

- Runs n in: 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000 (override via --ns)
- Times CPU and GPU runs (skips GPU if unavailable or errors)
- Computes speedup = CPU_time / GPU_time
- Saves CSV and PNG plot

Usage:
  python bench_intersection_test.py
  python bench_intersection_test.py --ns 10 100 1000 --repeats 3
  python bench_intersection_test.py --max-mem-gb 8  # skip cases estimated > 8 GB RAM
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from intersection_test import test
import matplotlib.pyplot as plt


@dataclass
class RunResult:
    n: int
    cpu_time: float
    gpu_time: float
    speedup: float
    cpu_ok: bool
    gpu_ok: bool
    note: str = ""


def estimate_bytes_for_n(n: int) -> int:
    """
    Estimate memory allocated INSIDE test() based on the arrays in the given code.

    Arrays (float32 = 4 bytes):
      proj_faces:      (n, 4, 3)   ->  n * 12 * 4
      tri_normal:      (n, 3)      ->  n * 3 * 4
      tri_plane_norm:  (n, 3, 3)   ->  n * 9 * 4
      tri_vertices:    (n, 3, 3)   ->  n * 9 * 4
      tri_epsilon:     (n,)        ->  n * 1 * 4
      computed_area:   (n,)        ->  n * 1 * 4
    Total bytes per n = (12 + 3 + 9 + 9 + 1 + 1) * 4 = 35 * 4 = 140 bytes
    """
    return int(n * 140)


def fmt_bytes(num):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"


def time_once(n: int, gpu: bool, timeout: float = None) -> tuple[bool, float, str]:
    """
    Time a single call to test(n, gpu=...).
    Returns: (ok, elapsed_sec or None, note)
    """
    t0 = time.perf_counter()
    try:
        # Optionally allow a crude timeout by polling time in a loop (simple approach).
        # Here we just execute directly; if you need hard timeouts, use multiprocessing.
        test(n=n, gpu=gpu)
        elapsed = time.perf_counter() - t0
        return True, elapsed, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def median(vals):
    s = sorted(vals)
    m = len(s)
    if m == 0:
        return math.nan
    if m % 2 == 1:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ns",
        nargs="*",
        type=float,
        default=[10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000],
        help="List of n values. Floats ending with .0e? are allowed; they will be cast to int.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per mode; takes median time.")
    parser.add_argument("--warmup-n", type=int, default=10_000, help="GPU warmup n (jit/alloc). 0 to disable.")
    parser.add_argument("--max-mem-gb", type=float, default=64.0, help="Skip n if estimated memory exceeds this.")
    parser.add_argument("--out-prefix", type=str, default="bench_intersection", help="Output file prefix.")
    args = parser.parse_args()

    ns = [int(n) for n in args.ns]

    # Warmup GPU once (helps JIT/first allocation)
    if args.warmup_n > 0:
        print(f"[warmup] GPU warmup with n={args.warmup_n}")
        ok, t, note = time_once(args.warmup_n, gpu=True)
        if ok:
            print(f"[warmup] GPU warmup done in {t:.4f} s")
        else:
            print(f"[warmup] GPU warmup failed ({note}). GPU runs may fail/compile later.")

    results: list[RunResult] = []

    for n in ns:
        est_bytes = estimate_bytes_for_n(n)
        est_gb = est_bytes / (1024**3)

        if est_gb > args.max_mem_gb:
            print(f"[skip] n={n:,}: estimated memory {fmt_bytes(est_bytes)} exceeds --max-mem-gb={args.max_mem_gb} GB")
            results.append(RunResult(n=n, cpu_time=None, gpu_time=None, speedup=None, cpu_ok=False, gpu_ok=False,
                                     note=f"Skipped due to estimated memory {fmt_bytes(est_bytes)}"))
            continue

        print(f"\n=== n={n:,} | est. alloc ~ {fmt_bytes(est_bytes)} ===")

        # CPU timing
        cpu_times = []
        cpu_note = ""
        cpu_ok = True
        for r in range(args.repeats):
            ok, t, note = time_once(n, gpu=False)
            if not ok:
                cpu_ok = False
                cpu_note = note
                break
            cpu_times.append(t)
            print(f"  CPU run {r+1}/{args.repeats}: {t:.4f} s")

        cpu_time = median(cpu_times) if cpu_ok else None
        if cpu_ok:
            print(f"  CPU median: {cpu_time:.4f} s")
        else:
            print(f"  CPU FAILED: {cpu_note}")

        # GPU timing
        gpu_times = []
        gpu_note = ""
        gpu_ok = True
        for r in range(args.repeats):
            ok, t, note = time_once(n, gpu=True)
            if not ok:
                gpu_ok = False
                gpu_note = note
                break
            gpu_times.append(t)
            print(f"  GPU run {r+1}/{args.repeats}: {t:.4f} s")

        gpu_time = median(gpu_times) if gpu_ok else None
        if gpu_ok:
            print(f"  GPU median: {gpu_time:.4f} s")
        else:
            print(f"  GPU FAILED: {gpu_note}")

        speed = (cpu_time / gpu_time) if (cpu_ok and gpu_ok and gpu_time and gpu_time > 0) else None
        note_parts = []
        if not cpu_ok:
            note_parts.append(f"CPU error: {cpu_note}")
        if not gpu_ok:
            note_parts.append(f"GPU error: {gpu_note}")
        results.append(RunResult(n=n, cpu_time=cpu_time, gpu_time=gpu_time, speedup=speed,
                                 cpu_ok=cpu_ok, gpu_ok=gpu_ok, note=" | ".join(note_parts)))

    # Save CSV
    csv_path = f"{args.out_prefix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "cpu_time_s", "gpu_time_s", "speedup_cpu_over_gpu", "cpu_ok", "gpu_ok", "note"])
        for r in results:
            w.writerow([r.n, r.cpu_time if r.cpu_time is not None else "",
                        r.gpu_time if r.gpu_time is not None else "",
                        r.speedup if r.speedup is not None else "",
                        int(r.cpu_ok), int(r.gpu_ok), r.note])
    print(f"\nSaved results to {csv_path}")

    # Plot: speedup vs n (log-x); only points where both runs succeeded
    xs = [r.n for r in results if r.speedup is not None]
    ys = [r.speedup for r in results if r.speedup is not None]

    if xs:
        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, marker="o")
        plt.xscale("log")
        plt.xlabel("n (log scale)")
        plt.ylabel("Speedup (CPU_time / GPU_time)")
        plt.title("Intersection Test: Speedup vs n")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        png_path = f"{args.out_prefix}_speedup.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"Saved plot to {png_path}")
        # Optionally show interactive window:
        # plt.show()
    else:
        print("No valid speedup points to plot (GPU or CPU runs failed/skipped).")


if __name__ == "__main__":
    main()
