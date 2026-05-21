import pytest
import torch

import flag_gems

from .hopper_fa3_utils import (
    accuracy_shapes,
    build_reference,
    dispatch_source,
    dispatches_to_hopper,
    is_fa3_supported,
    make_varlen,
    max_mean_abs,
    output_tensor,
    run_flag_gems,
    tolerances,
)


@pytest.mark.hopper_fa3
@pytest.mark.skipif(
    not is_fa3_supported(),
    reason="requires CUDA Hopper with Triton FA3 support",
)
def test_hopper_fa3_dispatch_installed():
    assert dispatches_to_hopper(), (
        "flag_gems.flash_attn_varlen_func is not routed to the Hopper backend; "
        f"source={dispatch_source()}"
    )


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(
    not is_fa3_supported(),
    reason="requires CUDA Hopper with Triton FA3 support",
)
@pytest.mark.parametrize("shape", accuracy_shapes(), ids=lambda shape: shape.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_hopper_fa3_varlen_accuracy(shape, dtype):
    tensors = make_varlen(shape, dtype, flag_gems.device, seed=2026)
    ref, ref_kind = build_reference(tensors, shape, fa_version=3)
    out = output_tensor(run_flag_gems(tensors, shape, fa_version=3))
    atol, rtol = tolerances(dtype, tensors.max_seqlen_k, ref_kind)

    max_abs, mean_abs = max_mean_abs(out, ref)
    msg = (
        f"shape={shape.name}, dtype={dtype}, ref={ref_kind}, "
        f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol, msg=msg)
