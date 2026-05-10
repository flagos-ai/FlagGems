import pytest
import torch

import flag_gems
from . import base, consts, utils


class SmoothL1LossBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)


def smooth_l1_loss_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    yield inp, target, 1, 1.0
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, target, 0, 1.0
        yield inp, target, 2, 1.0


def _smooth_l1_loss_out_baseline(inp, target, reduction, beta):
    """Baseline: functional + copy, avoids native out Resize warning."""
    loss = torch.ops.aten.smooth_l1_loss(inp, target, reduction, beta)
    out = torch.empty_like(loss)
    out.copy_(loss)
    return out


def _smooth_l1_loss_out_gems(inp, target, reduction, beta):
    """Gems: actual out variant under use_gems — optimized direct-write path."""
    out_shape = inp.shape if reduction == 0 else ()
    out = torch.empty(0, dtype=inp.dtype, device=inp.device)
    out.resize_(out_shape)
    with flag_gems.use_gems():
        return torch.ops.aten.smooth_l1_loss.out(inp, target, reduction, beta, out=out)


def smooth_l1_loss_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    yield inp, target, 1, 1.0
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, target, 0, 1.0
        yield inp, target, 2, 1.0


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss():
    bench = SmoothL1LossBenchmark(
        op_name="smooth_l1_loss",
        input_fn=smooth_l1_loss_input_fn,
        torch_op=torch.ops.aten.smooth_l1_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.smooth_l1_loss_out
def test_smooth_l1_loss_out():
    bench = SmoothL1LossBenchmark(
        op_name="smooth_l1_loss_out",
        input_fn=smooth_l1_loss_out_input_fn,
        torch_op=_smooth_l1_loss_out_baseline,
        gems_op=_smooth_l1_loss_out_gems,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
