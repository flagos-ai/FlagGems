# HINGE_EMBEDDING_LOSS operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.hinge_embedding_loss import (
    hinge_embedding_loss as gems_hinge_embedding_loss,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.hinge_embedding_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hinge_embedding_loss_defaults(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target_tensor = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(
        dtype
    )

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()

    ref_out = torch.ops.aten.hinge_embedding_loss(ref_self, ref_target)

    with flag_gems.use_gems():
        act_out = gems_hinge_embedding_loss(self_tensor, target_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hinge_embedding_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("margin", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_hinge_embedding_loss_fullargs(shape, dtype, margin, reduction):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target_tensor = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(
        dtype
    )

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()

    ref_out = torch.ops.aten.hinge_embedding_loss(
        ref_self, ref_target, float(margin), int(reduction)
    )

    with flag_gems.use_gems():
        act_out = gems_hinge_embedding_loss(
            self_tensor, target_tensor, float(margin), int(reduction)
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hinge_embedding_loss
def test_perf_aten_hinge_embedding_loss():
    # Define input generation logic matching the operator arguments
    def hinge_embedding_loss_input_fn(shape, dtype, device):
        self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        target_tensor = (
            torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1
        ).to(dtype)
        yield self_tensor, target_tensor

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=hinge_embedding_loss_input_fn,
        op_name="hinge_embedding_loss",
        torch_op=torch.ops.aten.hinge_embedding_loss,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
