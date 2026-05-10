import pytest
import torch

import flag_gems
from . import base, consts, utils


class SmoothL1LossBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)


def smooth_l1_loss_backward_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    grad_out = utils.generate_tensor_input(shape, dtype, device)
    yield grad_out, inp, target, 1, 1.0
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield grad_out, inp, target, 0, 1.0
        yield grad_out, inp, target, 2, 1.0


def _smooth_l1_loss_backward_out_baseline(grad_output, inp, target, reduction, beta):
    """Baseline: functional + copy, avoids native out Resize warning."""
    result = torch.ops.aten.smooth_l1_loss_backward.default(
        grad_output, inp, target, reduction, beta
    )
    grad_input = torch.empty_like(inp)
    grad_input.copy_(result)
    return grad_input


def _smooth_l1_loss_backward_out_gems(grad_output, inp, target, reduction, beta):
    """Gems: actual out variant under use_gems — optimized direct-write path."""
    grad_input = torch.empty(0, dtype=inp.dtype, device=inp.device)
    grad_input.resize_(inp.shape)
    with flag_gems.use_gems():
        return torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, inp, target, reduction, beta, grad_input=grad_input
        )


def smooth_l1_loss_backward_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    grad_out = utils.generate_tensor_input(shape, dtype, device)
    yield grad_out, inp, target, 1, 1.0
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield grad_out, inp, target, 0, 1.0
        yield grad_out, inp, target, 2, 1.0


@pytest.mark.smooth_l1_loss_backward
def test_smooth_l1_loss_backward():
    bench = SmoothL1LossBackwardBenchmark(
        op_name="smooth_l1_loss_backward",
        input_fn=smooth_l1_loss_backward_input_fn,
        torch_op=torch.ops.aten.smooth_l1_loss_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.smooth_l1_loss_backward_out
def test_smooth_l1_loss_backward_out():
    bench = SmoothL1LossBackwardBenchmark(
        op_name="smooth_l1_loss_backward_out",
        input_fn=smooth_l1_loss_backward_out_input_fn,
        torch_op=_smooth_l1_loss_backward_out_baseline,
        gems_op=_smooth_l1_loss_backward_out_gems,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
