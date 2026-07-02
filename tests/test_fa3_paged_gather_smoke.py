import pytest
import torch

import flag_gems
from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.planning import (
    fa3_tle_metadata_dispatch,
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


def _skip_unless_hopper_fa3(pytestconfig) -> None:
    if pytestconfig.getoption("flash_attn_varlen_fa_version") != 3:
        pytest.skip("Hopper FA3 smoke only runs with fa_version=3.")
    if not is_fa3_supported():
        pytest.skip("requires CUDA Hopper with TLE FA3 support.")


_PAGED_SMOKE_SHAPES = [
    Shape(
        "paged_decode_q1_flashdecoding",
        [(1, 128), (1, 192), (1, 256)],
        8,
        2,
        128,
        True,
        paged=True,
        block_size=16,
    ),
    Shape(
        "paged_decodeish_q16_longk_flashdecoding",
        [(16, 1024)] + [(1, 512)] * 15,
        8,
        2,
        128,
        True,
        paged=True,
        block_size=16,
    ),
    Shape(
        "paged_short_blockwise",
        [(64, 64), (32, 96), (1, 128)],
        8,
        2,
        128,
        True,
        paged=True,
        block_size=16,
    ),
    Shape(
        "paged_benchmark_mixed_short",
        [(1, 1328), (5, 18), (129, 463)],
        8,
        2,
        128,
        True,
        paged=True,
        block_size=32,
    ),
]


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    "max_q,is_paged,has_cache_kv,num_splits,expected_mode,expected_split",
    [
        (4, True, True, 4, "splitkv_decode", True),
        (16, True, True, 1, "multi_token_decode", False),
        (64, True, True, 0, "multi_token_decode", False),
        (129, True, False, 0, "prefill", False),
    ],
    ids=[
        "q1-gqa-swapped-flashdecoding",
        "q16-longk-flashdecoding",
        "short-paged",
        "benchmark-mixed-short",
    ],
)
def test_fa3_paged_plan_smoke(
    max_q,
    is_paged,
    has_cache_kv,
    num_splits,
    expected_mode,
    expected_split,
):
    route = fa3_tle_metadata_dispatch(
        max_query_len=max_q,
        is_paged=is_paged,
        has_cache_kv=has_cache_kv,
        num_splits=num_splits,
        has_scheduler_metadata=num_splits > 0,
    )
    assert route.mode == expected_mode
    assert route.layout == "paged"
    assert route.split_kv == expected_split


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("paged_gather", ["auto", "blockwise", "legacy"])
@pytest.mark.parametrize("shape", _PAGED_SMOKE_SHAPES, ids=lambda shape: shape.name)
@torch.inference_mode()
def test_fa3_paged_gather_correctness_smoke(
    monkeypatch, pytestconfig, paged_gather, shape
):
    _skip_unless_hopper_fa3(pytestconfig)
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", paged_gather)

    tensors = make_varlen(shape, torch.float16, flag_gems.device, seed=2041)
    ref, ref_kind = build_reference(tensors, shape, fa_version=3)
    out = output_tensor(run_flag_gems(tensors, shape, fa_version=3))
    atol, rtol = tolerances(torch.float16, tensors.max_seqlen_k, ref_kind)
    max_abs, mean_abs = max_mean_abs(out, ref)
    msg = (
        f"shape={shape.name}, paged_gather={paged_gather}, ref={ref_kind}, "
        f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol, msg=msg)
