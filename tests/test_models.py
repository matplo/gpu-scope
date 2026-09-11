from gpu_scope.models import GpuSnapshot


def _snapshot(**overrides) -> GpuSnapshot:
    defaults = dict(
        index=0,
        uuid="gpu-0",
        name="Test GPU",
        vendor="NVIDIA",
        utilization_pct=50.0,
        memory_used_bytes=4 * 1024**3,
        memory_total_bytes=8 * 1024**3,
    )
    defaults.update(overrides)
    return GpuSnapshot(**defaults)


def test_memory_used_pct():
    snap = _snapshot()
    assert snap.memory_used_pct == 50.0


def test_memory_used_pct_missing_total_is_none():
    snap = _snapshot(memory_total_bytes=None)
    assert snap.memory_used_pct is None


def test_memory_used_pct_missing_used_is_none():
    snap = _snapshot(memory_used_bytes=None)
    assert snap.memory_used_pct is None
