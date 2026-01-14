# MARGIN_RANKING_LOSS operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.margin_ranking_loss import (
    margin_ranking_loss as gems_margin_ranking_loss,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import TO_CPU, gems_assert_close


except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.margin_ranking_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("margin", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_margin_ranking_loss_tensor(shape, dtype, margin, reduction):
    input1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    input2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (
        torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2 - 1
    ).to(dtype)

    ref_input1 = to_reference(input1)
    ref_input2 = to_reference(input2)
    ref_target = to_reference(target)

    ref_out = torch.ops.aten.margin_ranking_loss(
        ref_input1, ref_input2, ref_target, margin, reduction
    )

    with flag_gems.use_gems():
        act_out = gems_margin_ranking_loss(input1, input2, target, margin, reduction)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.margin_ranking_loss
def test_perf_aten_margin_ranking_loss():
    # Define input generation logic matching the operator arguments
    def margin_ranking_loss_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        target = (
            torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2
            - 1
        ).to(dtype)
        margin = 0.5  # Example margin
        reduction = 1  # Example reduction (mean)
        yield inp1, inp2, target, margin, reduction

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=margin_ranking_loss_input_fn,
        op_name="margin_ranking_loss",
        torch_op=torch.ops.aten.margin_ranking_loss,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
