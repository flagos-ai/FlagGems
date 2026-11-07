"""Real family router cache, graph-capture and dynamic operand contracts."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def router(monkeypatch):
    module = importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm_family")
    module.clear_mm_family_choices()
    monkeypatch.setattr(module.torch_device_fn, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield module
    module.clear_mm_family_choices()


def tensor(shape, stride, pointer, dtype=torch.bfloat16):
    return SimpleNamespace(
        ndim=2,
        shape=shape,
        stride=lambda: stride,
        dtype=dtype,
        device=torch.device("cuda:0"),
        data_ptr=lambda: pointer,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: pointer),
        requires_grad=False,
        is_inference=lambda: False,
    )


def operands(offset=0, dtype=torch.bfloat16):
    return (
        tensor((35, 512), (512, 1), 0x100000 + offset, dtype),
        tensor((512, 80), (80, 1), 0x100000000 + offset, dtype),
        tensor((35, 80), (80, 1), 0x200000000 + offset, dtype),
    )


def install_benchmark(monkeypatch):
    graph = importlib.import_module("triton.flagtune.runtime.graph_benchmark")
    measured = []

    def bench(fn, **kwargs):
        measured.append(kwargs)
        return float(fn())

    monkeypatch.setattr(graph, "do_bench_hcu_graph", bench)
    return measured


def test_family_hot_cache_uses_current_tensors_and_skips_pool_rebuild(
    router, monkeypatch
):
    measured = install_benchmark(monkeypatch)
    cfg = router.SharedMMConfig((64, 128, 32))
    enumerated, launched = [], []

    def candidates(*args, **kwargs):
        enumerated.append(args)
        return (cfg,)

    monkeypatch.setattr(router, "mm_family_candidates", candidates)

    def launch(a, b, c, config):
        launched.append((a, b, c, config))
        return 1

    monkeypatch.setattr(router, "_launch", launch)
    a, b, c = operands()
    assert router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    first_measurements = len(measured)
    new = operands(offset=16)
    assert router.try_mm_family(*new, lambda: 10, tuning_key=("default",))
    assert len(enumerated) == 1 and len(measured) == first_measurements
    assert all(x is y for x, y in zip(launched[-1][:3], new))
    # A warm graph uses exactly the cached decision without running a tuner.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert router.try_mm_family(*new, lambda: 10, tuning_key=("default",))
    assert len(enumerated) == 1 and len(measured) == first_measurements


def test_family_cold_graph_never_tunes(router, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cold capture must not tune or enumerate kernels")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(router, "mm_family_candidates", forbidden)
    assert not router.try_mm_family(*operands(), forbidden, tuning_key=("default",))
    assert not router._choices


def test_family_separates_dtype_protocol_and_runtime_eligibility(router, monkeypatch):
    install_benchmark(monkeypatch)
    cfg = router.SharedMMConfig((64, 128, 32))
    enumerated = []

    def candidates(a, b, c, **kwargs):
        enumerated.append((a.dtype, kwargs))
        return (cfg,) if cfg.supports(a, b, c) else ()

    monkeypatch.setattr(router, "mm_family_candidates", candidates)
    monkeypatch.setattr(router, "_launch", lambda *args: 1)
    assert router.try_mm_family(*operands(), lambda: 10, tuning_key=("default",))
    assert router.try_mm_family(
        *operands(dtype=torch.float16), lambda: 10, tuning_key=("default",)
    )
    assert router.try_mm_family(*operands(), lambda: 10, tuning_key=("expanded",))
    assert len(router._choices) == 3
    a, b, c = operands()
    c.data_ptr = lambda: 0x200000008
    assert not router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    assert len(enumerated) == 4 and len(router._choices) == 3


def test_family_does_not_reuse_legacy_choice_when_storage_alias_changes(
    router, monkeypatch
):
    install_benchmark(monkeypatch)

    def candidates(a, b, c, **kwargs):
        return (
            ("legacy",)
            if c.untyped_storage().data_ptr() != a.untyped_storage().data_ptr()
            else ()
        )

    monkeypatch.setattr(router, "mm_family_candidates", candidates)
    monkeypatch.setattr(router, "_launch", lambda *args: 1)
    a, b, c = operands()
    assert router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    c.untyped_storage = a.untyped_storage
    assert not router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))


def test_family_does_not_reuse_choice_for_different_output_shape(router, monkeypatch):
    install_benchmark(monkeypatch)
    cfg = router.SharedMMConfig((64, 128, 32))
    monkeypatch.setattr(
        router,
        "mm_family_candidates",
        lambda a, b, c, **kwargs: (cfg,) if cfg.supports(a, b, c) else (),
    )
    monkeypatch.setattr(router, "_launch", lambda *args: 1)
    a, b, c = operands()
    assert router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    c.shape = (34, 80)
    assert not router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
