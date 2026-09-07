from __future__ import annotations

import os
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import psutil
import torch


@dataclass
class BenchmarkResult:
    repetitions: int
    warmup: int
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_min: float
    latency_ms_max: float
    throughput_samples_per_s: float
    process_rss_before_mib: float
    process_rss_after_mib: float
    process_rss_peak_mib: float
    cuda_max_allocated_mib: float | None
    cuda_max_reserved_mib: float | None
    gpu_energy_j: float | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RSSMonitor:
    """Samples process RSS without changing the function under measurement."""

    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self._process = psutil.Process()
        self.peak = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._process.memory_info().rss)
            time.sleep(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak = max(self.peak, self._process.memory_info().rss)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("No latency values were collected")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _sample_gpu_energy(device: torch.device, duration_s: float = 0.02) -> tuple[threading.Event, threading.Thread, list[tuple[float, float]]] | None:
    """Returns a lightweight NVML sampler, or None if unavailable.

    Energy is intentionally reported only when NVML exposes instantaneous GPU power.
    CPU energy is omitted because this tool has no reliable cross-platform package-power sensor.
    """
    if device.type != "cuda":
        return None
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
    except Exception:
        return None

    samples: list[tuple[float, float]] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0))
            except Exception:
                return
            time.sleep(duration_s)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return stop, thread, samples


def _integrate_energy(samples: list[tuple[float, float]]) -> float | None:
    if len(samples) < 2:
        return None
    joules = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        joules += (p0 + p1) * 0.5 * (t1 - t0)
    return joules


def benchmark_forward(
    forward: Callable[[torch.Tensor], Any],
    batch: torch.Tensor,
    *,
    device: torch.device,
    warmup: int = 20,
    repetitions: int = 100,
) -> BenchmarkResult:
    """Measures loaded-model inference; model loading and network I/O are excluded."""
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")

    process = psutil.Process()
    rss_before = process.memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    monitor = RSSMonitor()
    monitor.start()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = forward(batch)
            _sync(device)

            energy_sampler = _sample_gpu_energy(device)
            latencies_s: list[float] = []
            for _ in range(repetitions):
                _sync(device)
                start = time.perf_counter_ns()
                output = forward(batch)
                _sync(device)
                latencies_s.append((time.perf_counter_ns() - start) / 1e9)
                del output
            if energy_sampler is not None:
                stop, thread, samples = energy_sampler
                stop.set()
                thread.join(timeout=1)
                energy_j = _integrate_energy(samples)
            else:
                energy_j = None
    finally:
        monitor.stop()

    rss_after = process.memory_info().rss
    latencies_ms = [value * 1000.0 for value in latencies_s]
    batch_size = int(batch.shape[0])
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python_platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "batch_shape": list(batch.shape),
        "dtype": str(batch.dtype).replace("torch.", ""),
        "measurement_scope": "loaded-model forward only; excludes model loading, network download, probe fitting, and disk I/O",
    }
    return BenchmarkResult(
        repetitions=repetitions,
        warmup=warmup,
        latency_ms_mean=statistics.fmean(latencies_ms),
        latency_ms_median=statistics.median(latencies_ms),
        latency_ms_p50=_percentile(latencies_ms, 50),
        latency_ms_p95=_percentile(latencies_ms, 95),
        latency_ms_min=min(latencies_ms),
        latency_ms_max=max(latencies_ms),
        throughput_samples_per_s=(batch_size * repetitions) / sum(latencies_s),
        process_rss_before_mib=rss_before / (1024**2),
        process_rss_after_mib=rss_after / (1024**2),
        process_rss_peak_mib=monitor.peak / (1024**2),
        cuda_max_allocated_mib=(torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None),
        cuda_max_reserved_mib=(torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None),
        gpu_energy_j=energy_j,
        metadata=metadata,
    )
