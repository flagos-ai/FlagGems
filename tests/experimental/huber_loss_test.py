# HUBER_LOSS operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.huber_loss import huber_loss as gems_huber_loss
from flag_gems.experimental_ops.huber_loss import huber_loss_out as gems_huber_loss_out

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.huber_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_tensor(shape, dtype, reduction, delta):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()
    ref_out = torch.ops.aten.huber_loss(ref_self, ref_target, reduction, float(delta))

    with flag_gems.use_gems():
        act_self = self_tensor.clone()
        act_target = target_tensor.clone()
        act_out = gems_huber_loss(act_self, act_target, reduction, float(delta))

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.huber_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_out(shape, dtype, reduction, delta):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if reduction == 0:
        out_shape = shape
    else:
        out_shape = ()

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()
    ref_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    torch.ops.aten.huber_loss.out(
        ref_self, ref_target, reduction, float(delta), out=ref_out
    )

    with flag_gems.use_gems():
        act_self = self_tensor.clone()
        act_target = target_tensor.clone()
        act_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        gems_huber_loss_out(act_self, act_target, reduction, float(delta), act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.huber_loss
def test_perf_aten_huber_loss():
    # Define input generation logic matching the operator arguments
    def huber_loss_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        delta = 1.0  # You can choose a fixed delta for benchmarking
        reduction = 0  # You can choose a fixed reduction for benchmarking
        yield inp1, inp2, reduction, delta

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=huber_loss_input_fn,
        op_name="huber_loss",
        torch_op=torch.ops.aten.huber_loss,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
