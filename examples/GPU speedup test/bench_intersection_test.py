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

import numpy as np
from geometry import get_intersection_area
from geometry_gpu import get_intersection_area_gpu_profiler
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
from numba import cuda


@dataclass
class RunResult:
    n: int
    cpu_time: float
    gpu_call_time: float          # end-to-end time of Python call (includes host array creation)
    gpu_h2d_s: float           # measured inside geometry_gpu (H->D copies)
    gpu_alloc_s: float           # device allocations inside geometry_gpu
    gpu_kernel_ms: float         # kernel time from CUDA events (device-side)
    gpu_kernel_submit_s: float    # host-side time to enqueue the kernel
    gpu_kernel_call_s: float      # host-side enqueue + wait-for-completion
    gpu_kernel_launch_overhead_s: float  # approx: kernel_call_s - kernel_ms
    gpu_d2h_s: float              # measured inside geometry_gpu (D->H copy)
    gpu_total_s: float            # h2d + alloc + kernel_ms + d2h (inside geometry_gpu)
    gpu_total_incl_launch_s: float  # h2d + alloc + kernel_call_s + d2h
    speedup: float                # CPU_time / gpu_call_time (or choose /gpu_total_s)
    cpu_ok: bool
    gpu_ok: bool
    note: str = ""


def test_routine(n=1, gpu=False):
    proj_faces = np.tile(np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=np.float32), (n, 1, 1))
    tri_normal = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
    tri_plane_normal = np.tile(np.array([[[-1, 0, 0], [0, -1, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0]]], dtype=np.float32), (n, 1, 1))
    tri_vertices = np.tile(np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32), (n, 1, 1))
    tri_epsilon = np.full((n,), 1e-6, dtype=np.float32)

    if gpu:
        computed_area, prof = get_intersection_area_gpu_profiler(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)
        return computed_area, prof
    else:
        computed_area = get_intersection_area(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)
        return computed_area, None




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


def time_once(n: int, gpu: bool) -> tuple[bool, float, str]:
    """
    Time a single call to test(n, gpu=...).
    Returns: (ok, elapsed_sec or None, note)
    """
    t0 = time.perf_counter()
    try:
        # Optionally allow a crude timeout by polling time in a loop (simple approach).
        test_routine(n=n, gpu=gpu)
        elapsed = time.perf_counter() - t0
        return True, elapsed, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"
    
def time_once_gpu_profiled(n: int) -> tuple[bool, float, dict, str]:
    """
    Times a single GPU run and returns:
      (ok, gpu_call_elapsed_s, prof_dict, note)
    gpu_call_elapsed_s: full Python call time (includes host array creation in intersection_test)
    prof_dict: {'h2d_s','alloc_s','kernel_ms','d2h_s','total_s'} from geometry_gpu
    """
    try:
        # Warm GPU sync before starting
        cuda.synchronize()
        t0 = time.perf_counter()

        _, prof = test_routine(n=n, gpu=True)

        # prof["total_s"] already includes syncs inside geometry_gpu.py
        elapsed = time.perf_counter() - t0
        # elapsed includes host array creation + function overhead; prof["total_s"] is inside get_intersection_area_gpu()
        return True, elapsed, prof, ""
    except Exception as e:
        return False, None, {}, f"{type(e).__name__}: {e}"


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
        ok, t, prof, note = time_once_gpu_profiled(n=args.warmup_n)
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
            results.append(RunResult(
                n=n,
                cpu_time=None,
                gpu_call_time=None,
                gpu_h2d_s=None,
                gpu_alloc_s=None,
                gpu_kernel_ms=None,
                gpu_kernel_submit_s=None,
                gpu_kernel_call_s=None,
                gpu_kernel_launch_overhead_s=None,
                gpu_d2h_s=None,
                gpu_total_s=None,
                gpu_total_incl_launch_s=None,
                speedup=None,
                cpu_ok=False,
                gpu_ok=False,
                note=f"Skipped due to estimated memory {fmt_bytes(est_bytes)}",
            ))
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
        gpu_h2d = []
        gpu_alloc = []
        gpu_kernel_ms = []
        gpu_kernel_submit_s = []
        gpu_kernel_call_s = []
        gpu_kernel_launch_overhead_s = []
        gpu_d2h = []
        gpu_total = []
        gpu_total_incl_launch = []
        gpu_note = ""
        gpu_ok = True
        for r in range(args.repeats):
            ok, t, prof, note = time_once_gpu_profiled(n)
            if not ok:
                gpu_ok = False
                gpu_note = note
                break
            gpu_times.append(t)
            print(f"  GPU run {r+1}/{args.repeats}: {t:.4f} s")
            gpu_h2d.append(prof.get("h2d_s", float("nan")))
            gpu_alloc.append(prof.get("alloc_s", float("nan")))
            gpu_kernel_ms.append(prof.get("kernel_ms", float("nan")))
            gpu_kernel_submit_s.append(prof.get("kernel_submit_s", float("nan")))
            gpu_kernel_call_s.append(prof.get("kernel_call_s", float("nan")))
            gpu_kernel_launch_overhead_s.append(prof.get("kernel_launch_overhead_s", float("nan")))
            gpu_d2h.append(prof.get("d2h_s", float("nan")))
            gpu_total.append(prof.get("total_s", float("nan")))
            gpu_total_incl_launch.append(prof.get("total_incl_launch_s", float("nan")))

        if gpu_ok:
            gpu_call_time = sorted(gpu_times)[len(gpu_times)//2]
            gpu_h2d_s     = sorted(gpu_h2d)[len(gpu_h2d)//2]
            gpu_alloc_s   = sorted(gpu_alloc)[len(gpu_alloc)//2]
            gpu_kernel    = sorted(gpu_kernel_ms)[len(gpu_kernel_ms)//2]
            gpu_k_submit  = sorted(gpu_kernel_submit_s)[len(gpu_kernel_submit_s)//2]
            gpu_k_call    = sorted(gpu_kernel_call_s)[len(gpu_kernel_call_s)//2]
            gpu_k_over    = sorted(gpu_kernel_launch_overhead_s)[len(gpu_kernel_launch_overhead_s)//2]
            gpu_d2h_s     = sorted(gpu_d2h)[len(gpu_d2h)//2]
            gpu_total_s   = sorted(gpu_total)[len(gpu_total)//2]
            gpu_total_incl_launch_s = sorted(gpu_total_incl_launch)[len(gpu_total_incl_launch)//2]
        else:
            gpu_call_time = gpu_h2d_s = gpu_alloc_s = gpu_kernel = None
            gpu_k_submit = gpu_k_call = gpu_k_over = None
            gpu_d2h_s = gpu_total_s = gpu_total_incl_launch_s = None

        if gpu_ok:
            print(f"  GPU median: {gpu_call_time:.4f} s")
        else:
            print(f"  GPU FAILED: {gpu_note}")

        speed = (cpu_time / gpu_call_time) if (cpu_ok and gpu_ok and gpu_call_time and gpu_call_time > 0) else None
        note_parts = []
        if not cpu_ok:
            note_parts.append(f"CPU error: {cpu_note}")
        if not gpu_ok:
            note_parts.append(f"GPU error: {gpu_note}")
        results.append(RunResult(
            n=n,
            cpu_time=cpu_time,
            gpu_call_time=gpu_call_time,
            gpu_h2d_s=gpu_h2d_s,
            gpu_alloc_s=gpu_alloc_s,
            gpu_kernel_ms=gpu_kernel,
            gpu_kernel_submit_s=gpu_k_submit,
            gpu_kernel_call_s=gpu_k_call,
            gpu_kernel_launch_overhead_s=gpu_k_over,
            gpu_d2h_s=gpu_d2h_s,
            gpu_total_s=gpu_total_s,
            gpu_total_incl_launch_s=gpu_total_incl_launch_s,
            speedup=speed,
            cpu_ok=cpu_ok,
            gpu_ok=gpu_ok,
            note=" | ".join(note_parts),
        ))

    # Save CSV
    csv_path = f"{args.out_prefix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n",
            "cpu_time_s",
            "gpu_time_s",
            "gpu_h2d_s",
            "gpu_alloc_s",
            "gpu_kernel_ms",
            "gpu_kernel_submit_s",
            "gpu_kernel_call_s",
            "gpu_kernel_launch_overhead_s",
            "gpu_d2h_s",
            "gpu_total_s",
            "gpu_total_incl_launch_s",
            "speedup_cpu_over_gpu",
            "cpu_ok",
            "gpu_ok",
            "note",
        ])
        for r in results:
            w.writerow([r.n, r.cpu_time if r.cpu_time is not None else "",
                        r.gpu_call_time if r.gpu_call_time is not None else "",
                        r.gpu_h2d_s if r.gpu_h2d_s is not None else "",
                        r.gpu_alloc_s if r.gpu_alloc_s is not None else "",
                        r.gpu_kernel_ms if r.gpu_kernel_ms is not None else "",
                        r.gpu_kernel_submit_s if r.gpu_kernel_submit_s is not None else "",
                        r.gpu_kernel_call_s if r.gpu_kernel_call_s is not None else "",
                        r.gpu_kernel_launch_overhead_s if r.gpu_kernel_launch_overhead_s is not None else "",
                        r.gpu_d2h_s if r.gpu_d2h_s is not None else "",
                        r.gpu_total_s if r.gpu_total_s is not None else "",
                        r.gpu_total_incl_launch_s if r.gpu_total_incl_launch_s is not None else "",
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
