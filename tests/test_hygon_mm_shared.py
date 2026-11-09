"""Ordinary shared launch-policy boundary contracts."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def shared():
    return importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm_shared")


def _tensor(shape, stride, pointer):
    return SimpleNamespace(
        ndim=2,
        shape=shape,
        stride=lambda: stride,
        dtype=torch.bfloat16,
        device=SimpleNamespace(type="cuda"),
        data_ptr=lambda: pointer,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: pointer),
        requires_grad=False,
        is_inference=lambda: False,
    )


@pytest.mark.parametrize(
    "m,n,k,nt",
    [
        (35, 88, 128, False),
        (35, 80, 384, True),
        (35, 88, 8320, False),
        (17, 8200, 128, False),
        (32769, 32, 32768, True),
        (35, 80, 8, False),
        (35, 80, 24, True),
        (129, 136, 520, False),
    ],
)
def test_shared_policy_uses_vector_and_pipeline_bounds(shared, m, n, k, nt):
    config = shared.SharedMMConfig((64, 128, 32))
    a, b, c = (
        _tensor((m, k), (k, 1), 0x100000),
        _tensor((k, n), (1, k) if nt else (n, 1), 0x100000000),
        _tensor((m, n), (n, 1), 0x200000000),
    )
    assert config.supports(a, b, c)
    c.data_ptr = lambda: 50
    assert not config.supports(a, b, c)


def test_shared_policy_rejects_unaligned_k_and_alias(shared):
    config = shared.SharedMMConfig((64, 128, 32))
    a, b, c = (
        _tensor((35, 95), (95, 1), 0x100000),
        _tensor((95, 88), (88, 1), 0x100000000),
        _tensor((35, 88), (88, 1), 0x200000000),
    )
    assert not config.supports(a, b, c)
    a.shape, b.shape = (35, 128), (128, 88)
    a.stride = lambda: (128, 1)
    assert config.supports(a, b, c)
    c.data_ptr = a.data_ptr
    assert not config.supports(a, b, c)


def test_shared_partial_wave_tiles_are_compiler_choices(shared):
    shared.SharedMMConfig((32, 64, 32), warps=(2, 2))


def test_shared_backend_options_are_not_mutated_across_launches(shared):
    config = shared.SharedMMConfig((256, 256, 16), warps=(4, 2))
    first = dict(shared._compile_parameters(config, False))
    assert "post_mma_wait" not in first and "mma_phase_schedule" not in first
    assert first["mmac_warps_m"] == 4 and first["mmac_transposed"]
    assert first["prefer_wide_async_copy"] and first["USE_TLE"]
    assert first["kpack"] == 2
    assert "A_SHARED_LAYOUT" not in first and "B_SHARED_LAYOUT" not in first
    assert "USE_MATRIX_COPY" not in first and "TILES_PER_WARP" not in first
    assert (
        "A_SOURCE_BASIS" not in first
        and "ENTRY_FAMILY" not in first
        and "WAIT_MODE" not in first
    )
    first.pop("num_warps")
    assert dict(shared._compile_parameters(config, False))["num_warps"] == 8


def test_shared_policy_allows_disjoint_views_of_one_storage(shared):
    config = shared.SharedMMConfig((64, 128, 32))
    a = _tensor((35, 128), (128, 1), 0x100000)
    b = _tensor((128, 88), (88, 1), 0x100000000)
    c = _tensor((35, 88), (88, 1), 0x100000 + 2 * 35 * 128)
    c.untyped_storage = a.untyped_storage
    assert config.supports(a, b, c)
    c.data_ptr = lambda: 0x100000 + 2 * 35 * 128 - 16
    assert not config.supports(a, b, c)


def test_shared_policy_keeps_the_actual_local_resource_bound(shared):
    config = shared.SharedMMConfig((64, 128, 32))
    m, n, k = 1, 1, 2**25
    a = _tensor((m, k), (k, 1), 0x100000)
    b = _tensor((k, n), (1, k), 0x100000000)
    c = _tensor((m, n), (n, 1), 0x200000000)
    # A full tile span need not fit when the whole global array permits a
    # buffer-offset lowering. Instruction selection belongs to the compiler.
    assert config.supports(a, b, c)


@pytest.mark.parametrize(
    "tile,warps",
    [
        ((16, 32, 32), (1, 1)),
        ((32, 64, 32), (1, 2)),
        ((128, 64, 32), (2, 2)),
        ((64, 128, 32), (2, 2)),
    ],
)
def test_shared_config_has_no_old_wave_or_register_tile_filter(shared, tile, warps):
    shared.SharedMMConfig(tile, warps=warps)


def test_shared_pool_is_one_registry_for_both_input_layouts(shared):
    configs = shared.shared_mm_configs()
    assert configs and len(configs) == len(set(configs))
    for nt in (False, True):
        a = _tensor((35, 512), (512, 1), 0x100000)
        b = _tensor((512, 80), (1, 512) if nt else (80, 1), 0x100000000)
        c = _tensor((35, 80), (80, 1), 0x200000000)
        assert all(config.supports(a, b, c) for config in configs)
    assert len(configs) == 12
    assert set(configs[0].__dataclass_fields__) == {
        "tile",
        "warps",
        "group_m",
        "address_mode",
    }


@pytest.mark.parametrize(
    "m,n,k,nt,narrow",
    [
        (136, 2048, 4096, False, True),
        (1, 2**29, 128, True, False),
        (32769, 32, 32768, True, True),
    ],
)
def test_shared_launch_index_proof_includes_padded_output(
    shared, monkeypatch, m, n, k, nt, narrow
):
    # A large global input does not require wide tile coordinates, whereas
    # masked output rows can overflow even when the logical output fits.
    config = shared.SharedMMConfig((64, 128, 32))
    a = _tensor((m, k), (k, 1), 1 << 20)
    b = _tensor((k, n), (1, k) if nt else (n, 1), 1 << 46)
    c = _tensor((m, n), (n, 1), 1 << 40)
    calls = []

    class Entry:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((args, kwargs))
                return None, {}

            return launch

    monkeypatch.setattr(shared, "_ordinary_mm_entry", lambda: Entry())
    monkeypatch.setattr(shared.torch_device_fn, "device", lambda _: nullcontext())
    shared.launch_shared_mm(a, b, c, config)
    args, kwargs = calls[0]
    assert kwargs["USE_I32_INDEX"] is narrow
    assert kwargs["B_K_CONTIGUOUS"] is nt
    assert tuple(x.value for x in args[3:6]) == (m, n, k)
