"""Ordinary shared-chart legality and real launch-policy boundary contracts."""

import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def shared():
    pytest.importorskip("triton.experimental.gluon")
    gl = importlib.import_module("triton.experimental.gluon.language")
    if not hasattr(gl, "SharedLinearLayout"):
        pytest.skip("ordinary shared linear layouts unavailable")
    return importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm_shared")


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (128, 16), (16, 256)])
@pytest.mark.parametrize("minor", [0, 1])
@pytest.mark.parametrize("group", range(7))
def test_shared_charts_biject_and_preserve_full_vectors(shared, shape, minor, group):
    layouts = importlib.import_module(shared.__package__ + ".mm_shared_layout")
    layout = layouts.tile_shared_layout(
        shape, minor, layouts.row_bit_order(group), ((9, 3), (10, 4))
    )
    coords = []
    for offset in range(shape[0] * shape[1]):
        point = [0, 0]
        for bit, basis in enumerate(layout.offset_bases):
            if offset >> bit & 1:
                point = [x ^ y for x, y in zip(point, basis)]
        assert all(0 <= x < extent for x, extent in zip(point, shape))
        coords.append(tuple(point))
    assert len(set(coords)) == len(coords)
    for offset in range(0, len(coords), 8):
        assert [p[minor] for p in coords[offset : offset + 8]] == list(
            range(coords[offset][minor], coords[offset][minor] + 8)
        )
        assert len({p[1 - minor] for p in coords[offset : offset + 8]}) == 1


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
    assert replace(config, matrix_copy=False).supports(a, b, c) == (
        max(2 * m * k, 2 * k * n) < 2**31
    )
    c.data_ptr = lambda: 50
    assert not config.supports(a, b, c)


def test_shared_policy_rejects_invalid_pipeline_and_alias(shared):
    config = shared.SharedMMConfig((64, 128, 32))
    a, b, c = (
        _tensor((35, 96), (96, 1), 0x100000),
        _tensor((96, 88), (88, 1), 0x100000000),
        _tensor((35, 88), (88, 1), 0x200000000),
    )
    assert not config.supports(a, b, c)
    a.shape, b.shape = (35, 128), (128, 88)
    a.stride = lambda: (128, 1)
    assert config.supports(a, b, c)
    c.data_ptr = a.data_ptr
    assert not config.supports(a, b, c)


def test_shared_matrix_copy_requires_complete_active_waves(shared):
    with pytest.raises(ValueError, match="complete 16B"):
        shared.SharedMMConfig((32, 64, 32), warps=(2, 2))
    # Buffer copies are a separate ordinary strategy, not subject to the target
    # matrix-copy lowering's current all-wave requirement.
    shared.SharedMMConfig((32, 64, 32), warps=(2, 2), matrix_copy=False)


def test_shared_backend_options_are_not_mutated_across_launches(shared):
    config = shared.SharedMMConfig(
        (256, 256, 16), warps=(4, 2), row_group_bits=3, output_layout=2, post_mma=True
    )
    first = dict(shared._compile_parameters(config, False))
    assert first["post_mma_wait"] and first["mma_phase_schedule"]
    assert "mmac_warps_m" not in first and "mmac_transposed" not in first
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
    # Global arrays fit the byte-offset buffer strategy, but the current
    # matrix-copy box's full tile span would exceed signed31 bits.
    assert not config.supports(a, b, c)
    assert replace(config, matrix_copy=False).supports(a, b, c)


@pytest.mark.parametrize(
    "tile,warps,tiles",
    [
        ((16, 32, 32), (1, 1), (1, 1)),
        ((32, 64, 32), (1, 2), (1, 1)),
        ((128, 64, 32), (2, 2), (4, 1)),
        ((64, 128, 32), (2, 2), (1, 4)),
    ],
)
def test_shared_config_has_no_old_wave_or_register_tile_filter(
    shared, tile, warps, tiles
):
    shared.SharedMMConfig(tile, warps=warps, tiles_per_warp=tiles)


def test_shared_pool_is_one_registry_for_both_input_layouts(shared):
    configs = shared.shared_mm_configs()
    assert configs and len(configs) == len(set(configs))
    for nt in (False, True):
        a = _tensor((35, 512), (512, 1), 0x100000)
        b = _tensor((512, 80), (1, 512) if nt else (80, 1), 0x100000000)
        c = _tensor((35, 80), (80, 1), 0x200000000)
        assert all(config.supports(a, b, c) for config in configs)
    assert {config.matrix_copy for config in configs} == {False, True}
    assert {config.post_mma for config in configs} == {False, True}
