# SMOOTH_L1_LOSS operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.smooth_l1_loss import (  # noqa: E402
    smooth_l1_loss as gems_smooth_l1_loss,
)
from flag_gems.experimental_ops.smooth_l1_loss import (  # noqa: E402
    smooth_l1_loss_out as gems_smooth_l1_loss_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
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


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_loss_tensor(shape, dtype, reduction, beta):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_target = to_reference(target)

    ref_out = torch.ops.aten.smooth_l1_loss(ref_self, ref_target, reduction, beta)

    with flag_gems.use_gems():
        act_out = gems_smooth_l1_loss(self, target, reduction, beta)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_loss_out(shape, dtype, reduction, beta):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    out_shape = shape if reduction == 0 else ()

    ref_self = to_reference(self)
    ref_target = to_reference(target)
    ref_out = torch.empty(out_shape, dtype=dtype, device=ref_self.device)
    ref_out = torch.ops.aten.smooth_l1_loss.out(
        ref_self, ref_target, reduction, beta, out=ref_out
    )

    with flag_gems.use_gems():
        act_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        act_out = gems_smooth_l1_loss_out(self, target, reduction, beta, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.smooth_l1_loss
def test_perf_aten_smooth_l1_loss():
    # Define input generation logic matching the operator arguments
    def smooth_l1_loss_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=smooth_l1_loss_input_fn,
        op_name="smooth_l1_loss",
        torch_op=torch.ops.aten.smooth_l1_loss,
        dtypes=[torch.float32, torch.float16],
    )

    return bench.run()
