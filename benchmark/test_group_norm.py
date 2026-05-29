import math

import pytest
import torch

from . import base, consts


# TODO(Qiming): Extract this to a base class
class NormBenchmark(base.GenericBenchmark):
    # TODO: add new metric

    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def group_norm_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    channel = shape[1]
    weight = torch.randn(
        [
            channel,
        ],
        dtype=dtype,
        device=device,
    )
    bias = torch.randn(
        [
            channel,
        ],
        dtype=dtype,
        device=device,
    )
    yield inp, channel // 2, weight, bias

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, channel, weight, bias


@pytest.mark.group_norm
def test_group_norm():
    bench = NormBenchmark(
        input_fn=group_norm_input_fn,
        op_name="group_norm",
        torch_op=torch.nn.functional.group_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class NormBackwardBenchmark(NormBenchmark):
    """Limit shapes to avoid Triton compilation error: numel exceeds maximum (1048576)."""

    MAX_ELEMENTS = 2**20

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [s for s in shapes if math.prod(s) <= self.MAX_ELEMENTS]

    def init_user_config(self):
        super().init_user_config()
        self.shapes = [
            s for s in self.shapes if math.prod(s) <= self.MAX_ELEMENTS
        ]


@pytest.mark.group_norm_backward
def test_group_norm_backward():
    bench = NormBackwardBenchmark(
        input_fn=group_norm_input_fn,
        op_name="group_norm_backward",
        torch_op=torch.nn.functional.group_norm,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
