#!/usr/bin/env bash
# check_numba_cuda.sh
# Verify Numba and CUDA (for Numba's CUDA target)

set -u

echo "=== Numba + CUDA Installation Check ==="

# ---- Choose python ----
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "❌ Python not found in PATH."
  exit 1
fi

echo "Using: $($PYTHON --version) at $(command -v $PYTHON)"

# ---- Optional host-side CUDA tools (not strictly required by Numba) ----
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n1)"
else
  echo "ℹ️  nvidia-smi not found (ok, but useful for diagnostics)."
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc: $((nvcc --version | tail -n1) 2>/dev/null)"
else
  echo "ℹ️  nvcc not found (Numba does not require nvcc to run CUDA kernels)."
fi

# ---- Run Python diagnostics ----
"$PYTHON" - <<'PYCODE'
import sys, textwrap
ok = True

print("\n--- Python package checks ---")
try:
    import numba, numpy, llvmlite
    print(f"✅ Numba      : {numba.__version__}")
    print(f"✅ NumPy      : {numpy.__version__}")
    print(f"✅ llvmlite   : {llvmlite.__version__}")
except Exception as e:
    print(f"❌ Import failure (Numba/NumPy/llvmlite): {e}")
    sys.exit(1)

print("\n--- CUDA capability via Numba ---")
from numba import cuda
try:
    # Quick availability flag
    avail = cuda.is_available()
    print(f"CUDA available (numba.cuda.is_available): {avail}")
    if not avail:
        # Try to extract a more detailed reason
        try:
            cuda.detect()  # prints diagnostics if possible
        except Exception as e:
            print(f"Further CUDA detection info: {e}")
        raise SystemExit(1)

    # Versions: Driver & Runtime
    try:
        drv_ver = cuda.cudadrv.driver.get_version()
        print(f"✅ CUDA Driver version (from Numba): {drv_ver[0]}.{drv_ver[1]}")
    except Exception as e:
        print(f"⚠️  Could not query CUDA driver version: {e}")

    try:
        rt_ver = cuda.runtime.get_version()  # e.g., 12040 for 12.4
        if isinstance(rt_ver, int):
            major, minor = rt_ver // 1000, (rt_ver % 1000) // 10
            print(f"✅ CUDA Runtime version (from Numba): {major}.{minor}")
        else:
            print(f"✅ CUDA Runtime version (from Numba): {rt_ver}")
    except Exception as e:
        print(f"⚠️  Could not query CUDA runtime version: {e}")

    # List devices
    try:
        devs = list(cuda.gpus)
        if not devs:
            print("❌ No CUDA GPUs detected by Numba.")
            sys.exit(1)
        print("✅ Detected GPU(s):")
        for i, d in enumerate(devs):
            with d:
                name = d.name.decode() if isinstance(d.name, bytes) else d.name
                cc   = d.compute_capability
                print(f"   - GPU {i}: {name} (CC {cc[0]}.{cc[1]})")
    except Exception as e:
        print(f"❌ Error listing GPUs: {e}")
        sys.exit(1)

    # Tiny kernel test
    print("\n--- Running a tiny CUDA kernel test ---")
    import numpy as np

    @cuda.jit
    def add_kernel(a, b, c):
        i = cuda.grid(1)
        if i < a.size:
            c[i] = a[i] + b[i]

    N = 1 << 16
    a = np.ones(N, dtype=np.float32)
    b = np.full(N, 2.0, dtype=np.float32)
    c = np.zeros(N, dtype=np.float32)

    threads = 256
    blocks  = (N + threads - 1) // threads

    try:
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(c)
        add_kernel[blocks, threads](d_a, d_b, d_c)
        c_res = d_c.copy_to_host()
    except Exception as e:
        print(f"❌ CUDA kernel launch/copy failed: {e}")
        sys.exit(1)

    if np.allclose(c_res, a + b):
        print("✅ CUDA kernel ran successfully and results are correct.")
    else:
        print("❌ CUDA kernel ran but results are incorrect.")
        sys.exit(1)

except SystemExit as e:
    # Re-raise to propagate non-zero exit if we intentionally exited
    raise
except Exception as e:
    # Typically a CudaSupportError if drivers/toolkit mismatch or not visible
    print("❌ CUDA not usable via Numba.")
    print(textwrap.indent(str(e), prefix="   "))
    sys.exit(1)

print("\nAll checks passed ✅")
PYCODE

status=$?
echo
if [ $status -eq 0 ]; then
  echo "=== RESULT: SUCCESS ✅ Numba + CUDA are ready."
else
  echo "=== RESULT: FAILURE ❌ See messages above."
fi
exit $status
