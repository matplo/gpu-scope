from gpu_top.backends.demo import DemoBackend


def test_demo_backend_reports_requested_gpu_count():
    backend = DemoBackend(gpu_count=3)
    snap = backend.poll()
    assert snap.backend == "demo"
    assert len(snap.gpus) == 3
    assert [g.index for g in snap.gpus] == [0, 1, 2]


def test_demo_backend_values_are_in_range():
    backend = DemoBackend(gpu_count=1)
    snap = backend.poll()
    gpu = snap.gpus[0]
    assert 0.0 <= gpu.utilization_pct <= 100.0
    assert 0 <= gpu.memory_used_bytes <= gpu.memory_total_bytes
    assert gpu.memory_used_pct is not None


def test_demo_backend_is_always_available():
    assert DemoBackend.is_available() is True
