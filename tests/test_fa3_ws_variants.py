import os

import pytest
import torch

import flag_gems
from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.registry import (
    get_variant,
    variant_names,
)

from .hopper_fa3_utils import (
    Shape,
    build_reference,
    is_fa3_supported,
    make_varlen,
    max_mean_abs,
    output_tensor,
    run_flag_gems,
    tolerances,
)


def _shape_for_variant(name: str) -> Shape:
    spec = get_variant(name)
    if spec.shape_kind == "paged_decode":
        if spec.persistent:
            return Shape(
                f"{name}_paged_decode",
                [(1, 128)],
                4,
                4,
                128,
                True,
                paged=True,
                block_size=16,
            )
        return Shape(
            f"{name}_paged_decode",
            [(1, 128)],
            4,
            4,
            128,
            True,
            paged=True,
            block_size=16,
        )
    if spec.shape_kind == "small":
        return Shape(
            f"{name}_small",
            [(64, 64), (32, 96), (1, 128)],
            8,
            2,
            128,
            True,
        )
    if spec.persistent:
        return Shape(f"{name}_decode", [(1, 512)], 4, 4, 128, True)
    return Shape(f"{name}_decode", [(1, 1024)] * 4, 8, 2, 128, True)


def _install_variant_kernel(spec) -> None:
    from flag_gems.runtime.backend._nvidia.hopper.ops import flash_api_v3

    if spec.kernel_module == "fa_hopper_persistent_pingpong":
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws import (
            fa_hopper_persistent_pingpong,
        )

        flash_api_v3.flash_varlen_fwd_v3_tle_kernel = (
            fa_hopper_persistent_pingpong.flash_varlen_fwd_v3_tle_kernel
        )
        flash_api_v3.flash_varlen_fwd_v3_tle_ws_simple_kernel = (
            fa_hopper_persistent_pingpong.flash_varlen_fwd_v3_tle_ws_simple_kernel
        )
        return
    if spec.kernel_module == "fa_hopper_nonpersistent_tlx_style":
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws import (
            fa_hopper_nonpersistent_tlx_style,
        )

        flash_api_v3.flash_varlen_fwd_v3_tle_ws_short_kernel = (
            fa_hopper_nonpersistent_tlx_style.flash_varlen_fwd_v3_tle_ws_short_kernel
        )
        return
    raise AssertionError(f"unknown kernel module {spec.kernel_module}")


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("variant_name", variant_names())
@torch.inference_mode()
def test_fa3_ws_variant_smoke(monkeypatch, pytestconfig, variant_name):
    if pytestconfig.getoption("flash_attn_varlen_fa_version") != 3:
        pytest.skip("Hopper FA3 coverage only runs with fa_version=3.")
    if not is_fa3_supported():
        pytest.skip("requires CUDA Hopper with TLE FA3 support.")

    spec = get_variant(variant_name)
    _install_variant_kernel(spec)
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_FORCE_PATH", spec.force_path)
    if spec.paged:
        monkeypatch.setenv("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_D128", "1")
        monkeypatch.setenv("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL", "1")

    shape = _shape_for_variant(variant_name)
    tensors = make_varlen(shape, torch.float16, flag_gems.device, seed=2030)
    ref, ref_kind = build_reference(tensors, shape, fa_version=3)
    out = output_tensor(run_flag_gems(tensors, shape, fa_version=3))
    atol, rtol = tolerances(torch.float16, tensors.max_seqlen_k, ref_kind)
    max_abs, mean_abs = max_mean_abs(out, ref)
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=atol,
        rtol=rtol,
        msg=(
            f"variant={variant_name}, ref={ref_kind}, "
            f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
        ),
    )


def test_fa3_ws_registry_named_barrier_contract():
    # The TLE named-barrier API keeps arrive_count as the PTX total participant
    # thread count.  Two consumer warp groups therefore still use 256, not 512.
    source = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "flag_gems",
        "runtime",
        "backend",
        "_nvidia",
        "hopper",
        "ops",
        "flash_kernel_v3.py",
    )
    with open(source, encoding="utf-8") as f:
        text = f.read()
    assert "THREADS_IN_MMA_GROUPS: tl.constexpr = NUM_MMA_WARPS * 32" in text
    assert "arrive_count=THREADS_IN_MMA_GROUPS" in text
    assert "arrive_count=THREADS_IN_MMA_GROUPS * 2" not in text
