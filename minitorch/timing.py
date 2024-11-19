import minitorch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(
    backend: Union[minitorch.TensorBackend, minitorch.FastOps, minitorch.CudaOps],
    size: int = 16,
) -> None:
    """Run matrix multiplication using the specified backend and size."""
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    _ = x @ y


def plot_timings(times: dict, output_path: str = "timings_plot.png") -> None:
    """Plots the timings of the matrix multiplication for different sizes.
    It takes a dictionary of times and an output path for the plot as input.
    """
    sizes = list(times.keys())
    fast_times = [times[size]["fast"] for size in sizes]
    gpu_times = [times[size]["gpu"] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fast_times, marker="o", label="FastTensorBackend")
    plt.plot(sizes, gpu_times, marker="o", label="GPUBackend")
    plt.yscale("log")
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (s)")
    plt.title("Matrix Multiplication Timing: FastTensorBackend vs GPUBackend")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])
    plot_timings(times)

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")
