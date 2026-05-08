import pytest
import torch

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


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss():
    bench = SmoothL1LossBenchmark(
        op_name="smooth_l1_loss",
        input_fn=smooth_l1_loss_input_fn,
        torch_op=torch.ops.aten.smooth_l1_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
