"""Process CPU time and sampled resident memory; never Python-only allocations."""

import math
import threading
from statistics import fmean
from time import perf_counter

import psutil


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("Percentile requires samples and a probability in [0, 1]")
    values = sorted(values)
    position = (len(values) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def latency_summary(samples_ms: list[float]) -> dict:
    return {
        "samples": len(samples_ms),
        "p50_ms": percentile(samples_ms, 0.5),
        "p95_ms": percentile(samples_ms, 0.95),
        "mean_ms": fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


class ResourceMonitor:
    """A bounded sampler for this process, including its observer overhead.

    Sampled RSS can miss short peaks. CPU percent is one-core equivalent, not
    the Windows Task Manager percentage of the entire machine.
    """

    def __init__(self, interval_ms: int = 10):
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, int) or interval_ms < 1:
            raise ValueError("RSS sampling interval must be a positive integer")
        self.interval_ms = interval_ms
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak_rss = 0
        self.samples = 0
        self.error: Exception | None = None
        self.result: dict = {}

    def _sample(self):
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        self.samples += 1
        return rss

    def _loop(self):
        try:
            while not self.stop.wait(self.interval_ms / 1000):
                self._sample()
        except Exception as exc:
            self.error = exc

    def __enter__(self):
        self.rss_start = self._sample()
        self.cpu_start = self.process.cpu_times()
        self.started = perf_counter()
        self.thread = threading.Thread(target=self._loop, name="retrievallab-rss", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.thread.join()
        wall = perf_counter() - self.started
        cpu_end = self.process.cpu_times()
        cpu = (cpu_end.user - self.cpu_start.user) + (cpu_end.system - self.cpu_start.system)
        rss_end = self._sample()
        if self.error is not None and exc_type is None:
            raise RuntimeError("Resource measurement failed") from self.error
        self.result = {
            "wall_seconds": wall,
            "process_cpu_seconds": cpu,
            "cpu_percent_one_core_equivalent": 100 * cpu / wall if wall else 0.0,
            "rss_start_bytes": self.rss_start,
            "rss_end_bytes": rss_end,
            "sampled_peak_rss_bytes": self.peak_rss,
            "rss_samples": self.samples,
            "rss_sample_interval_ms": self.interval_ms,
            "scope": "whole process during stage, including sampler/runner overhead",
        }
