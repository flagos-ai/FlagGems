import pytest
import torch

from . import base, consts


class SmoothL1LossBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (262144,),
        (1024, 1024),
        (4096, 4096),
        (64, 512, 512),
    ]
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, device=self.device, dtype=dtype)
            target = torch.randn(shape, device=self.device, dtype=dtype)
            yield inp, target, 1, 1.0


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss():
    bench = SmoothL1LossBenchmark(
        op_name="smooth_l1_loss",
        torch_op=torch.ops.aten.smooth_l1_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
