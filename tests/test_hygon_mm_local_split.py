"""CTA-local split-K legality follows actual tile offsets, not model shapes."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def local():
    return importlib.import_module(
        "flag_gems.runtime.backend._hygon.ops.mm_local_split"
    )


def tensor(shape, stride, pointer, dtype):
    return SimpleNamespace(
        ndim=2,
        shape=shape,
        stride=lambda: stride,
        dtype=dtype,
        device=torch.device("cuda:0"),
        data_ptr=lambda: pointer,
        requires_grad=False,
        is_inference=lambda: False,
    )


def operands(m, n, k, dtype):
    return (
        tensor((m, k), (k, 1), 1 << 20, dtype),
        tensor((k, n), (1, k), 1 << 46, dtype),
        tensor((m, n), (n, 1), 1 << 47, dtype),
    )


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 1, 1), (1, 17, 13), (17, 1, 127), (19, 33, 129), (19, 33, 257), (19, 33, 769)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_partial_reduction_and_singleton_domains(local, m, n, k, dtype):
    for config in local.local_split_mm_configs():
        assert config.supports(*operands(m, n, k, dtype))


@pytest.mark.parametrize(
    "m,n,k", [(4194336, 1, 256), (1, 4194336, 256), (32769, 32769, 256)]
)
def test_global_2gib_is_not_local_buffer_limit(local, m, n, k):
    for config in local.local_split_mm_configs():
        assert config.supports(*operands(m, n, k, torch.bfloat16))


def test_global_offsets_are_i64_and_padded_coordinates_fit_i32(local):
    config = local.LocalSplitMMConfig()
    for shape in [(1, 1, 2**27), (1, 2**27, 256), (2**31 - 1, 1, 256)]:
        assert config.supports(*operands(*shape, torch.bfloat16))
    for shape in [(2**31 + 1, 1, 256), (1, 2**31 + 1, 256), (1, 1, 2**31 + 1)]:
        assert not config.supports(*operands(*shape, torch.bfloat16))


def test_alias_checks_actual_view_ranges(local):
    a, b, c = operands(17, 33, 13, torch.bfloat16)
    end = a.data_ptr() + 2 * 17 * 13
    c.data_ptr = lambda: (end + 15) // 16 * 16
    assert local.LocalSplitMMConfig().supports(a, b, c)
    c.data_ptr = lambda: a.data_ptr() + 16
    assert not local.LocalSplitMMConfig().supports(a, b, c)


def test_dtype_storage_and_output_contracts(local):
    c = local.LocalSplitMMConfig()
    assert not c.supports(*operands(19, 33, 257, torch.float32))
    a, b, out = operands(19, 33, 257, torch.bfloat16)
    b.stride = lambda: (33, 1)
    assert c.supports(a, b, out)
    b.stride = lambda: (66, 2)
    assert not c.supports(a, b, out)
    a, b, out = operands(19, 33, 257, torch.bfloat16)
    out.requires_grad = True
    assert not c.supports(a, b, out)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile": (8, 16)},
        {"k_step": 64},
        {"num_stages": True},
        {"num_stages": 3},
        {"m_tiles": 0},
        {"tile": (128, 64)},
        {"tile": (32, 64), "m_tiles": 3},
    ],
)
def test_invalid_resource_or_pipeline_combinations(local, kwargs):
    with pytest.raises(ValueError):
        local.LocalSplitMMConfig(**kwargs)


def test_config_identity_is_immutable(local):
    config = local.LocalSplitMMConfig([16, 32])
    assert config == local.LocalSplitMMConfig((16, 32))
    assert hash(config) == hash(local.LocalSplitMMConfig((16, 32)))


def test_no_model_specific_mv_pruner():
    module = importlib.import_module("flag_gems.runtime.backend._hygon.ops.mv")
    # This is a tuner contract: the actual wrapped tuner has no early filter
    # silently replacing its declared pool on two model reduction lengths.
    module.mv_kernel._apply_flagtune()
    assert module.mv_kernel.fn.early_config_prune is None


def test_singleton_stride_has_no_address_effect(local):
    a, b, c = operands(1, 1, 13, torch.bfloat16)
    a.stride = lambda: (0, 1)
    b.stride = lambda: (1, 1)
    c.stride = lambda: (1, 1)
    assert local.LocalSplitMMConfig().supports(a, b, c)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_natural_halfword_alignment(local, index):
    values = list(operands(19, 33, 257, torch.bfloat16))
    original = values[index].data_ptr()
    values[index].data_ptr = lambda: original + 2
    assert local.LocalSplitMMConfig().supports(*values)
    values[index].data_ptr = lambda: original + 1
    assert not local.LocalSplitMMConfig().supports(*values)


def test_pool_contains_logical_tiles_and_standard_pipeline_stages(local):
    configs = local.local_split_mm_configs()
    assert len(configs) == len(set(configs)) == 8
    assert set(configs[0].__dataclass_fields__) == {
        "tile",
        "k_step",
        "m_tiles",
        "num_stages",
    }
    assert {cfg.num_stages for cfg in configs} == {1, 2}
    assert local.LocalSplitMMConfig((64, 64)).m_tiles == 1
    assert local.LocalSplitMMConfig((16, 32), k_step=128, m_tiles=5).m_tiles == 5


def test_mv_mmac_cache_separates_dtype(local, monkeypatch):
    module = importlib.import_module(local.__package__ + ".mv_mmac")
    graph = importlib.import_module("triton.flagtune.runtime.graph_benchmark")
    monkeypatch.setattr(module, "_choices", {})
    monkeypatch.setattr(module, "_choice_measurements", {})
    monkeypatch.setattr(module, "_eligible", lambda *args: True)
    monkeypatch.setattr(module.torch_device_fn, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(module, "_launch_mmac", lambda a, b, out: out.fill_(1))
    calls = []

    def bench(fn, **kwargs):
        fn()
        calls.append(1)
        return 1.0

    monkeypatch.setattr(graph, "do_bench_hcu_graph", bench)
    for dtype in (torch.bfloat16, torch.float16, torch.bfloat16):
        a = torch.ones(3, 5, dtype=dtype)
        b = torch.ones(5, dtype=dtype)
        out = torch.empty(3, dtype=dtype)
        assert module.try_mv_mmac(
            a, b, out, lambda result: result.fill_(1), tuning_key=("test",)
        )
    assert len(calls) == 12
    assert len(module._choices) == 2
