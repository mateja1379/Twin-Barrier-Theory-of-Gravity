"""
env_check.py – Verify JAX sees the GPU and run a basic smoke test.

Usage:
    python env_check.py
"""
import time
import jax
import jax.numpy as jnp


def main():
    # ── Device info ──────────────────────────────────────────────
    devices = jax.devices()
    backend = jax.default_backend()
    print(f"JAX version : {jax.__version__}")
    print(f"Devices     : {devices}")
    print(f"Backend     : {backend}")

    if backend != "gpu":
        print("WARNING: JAX is NOT using GPU – falling back to CPU.")
        print("         Install jax[cuda12] and check CUDA drivers.")

    # ── Smoke test: large matmul on default device ───────────────
    N = 4096
    print(f"\nSmoke test: {N}x{N} matmul …")
    x = jnp.ones((N, N), dtype=jnp.float32)

    # warm-up (JIT compile)
    _ = (x @ x).block_until_ready()

    t0 = time.perf_counter()
    y = (x @ x).block_until_ready()
    dt = time.perf_counter() - t0

    print(f"  shape : {y.shape}")
    print(f"  sum   : {float(y.sum()):.1f}  (expected {N * N * N / N:.1f} = {float(N)**2:.1f})")
    print(f"  time  : {dt*1000:.2f} ms")
    print(f"\n{'='*50}")
    print("env_check PASSED" if backend == "gpu" else "env_check WARN (CPU fallback)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
