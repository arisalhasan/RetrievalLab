import pytest

from retrieval_lab.profiling import ResourceMonitor, latency_summary, percentile


def test_percentiles_use_linear_interpolation():
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)
    assert percentile([3], 0.5) == 3
    assert latency_summary([3, 1, 2])["p50_ms"] == 2


def test_resources_measure_process_memory_and_stop_the_sampler():
    with ResourceMonitor(1) as monitor:
        sum(range(10000))
    result = monitor.result
    assert result["rss_start_bytes"] > 0
    assert result["sampled_peak_rss_bytes"] >= result["rss_end_bytes"]
    assert result["rss_samples"] >= 2
    assert result["process_cpu_seconds"] >= 0
    assert result["wall_seconds"] > 0
    assert not monitor.thread.is_alive()


def test_sampler_is_stopped_when_the_stage_fails():
    with pytest.raises(ValueError, match="stage failed"), ResourceMonitor() as monitor:
        raise ValueError("stage failed")
    assert not monitor.thread.is_alive()
