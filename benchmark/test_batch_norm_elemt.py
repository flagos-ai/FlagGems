import pytest
import torch

from . import base, consts


class BatchNormElemtBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (16, 16, 64),
        (16, 16, 1024),
        (16, 16, 4098),
        (1, 8, 4, 4),
        (16, 8, 128, 128),
        (8, 128, 56, 56),
        (32, 512, 14, 14),
    ]

    def set_more_shapes(self):
        return [
            (2, 64, 112, 112),
            (64, 1024, 7, 7),
            (128, 256, 32, 32),
            (4, 32, 8, 8, 8),
        ]


def batch_norm_elemt_input_fn(shape, dtype, device):
    # batch_norm_elemt requires at least 2D input (N, C, ...)
    if len(shape) < 2:
        return
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    # PyTorch requires weight/bias/mean/invstd to be float32
    weight = torch.randn((C,), dtype=torch.float32, device=device)
    bias = torch.randn((C,), dtype=torch.float32, device=device)
    mean = torch.randn((C,), dtype=torch.float32, device=device)
    invstd = torch.randn((C,), dtype=torch.float32, device=device).abs() + 0.01
    eps = 0.0
    yield inp, weight, bias, mean, invstd, eps


def torch_batch_norm_elemt(inp, weight, bias, mean, invstd, eps):
    return torch.batch_norm_elemt(inp, weight, bias, mean, invstd, eps)


@pytest.mark.batch_norm_elemt
def test_batch_norm_elemt():
    bench = BatchNormElemtBenchmark(
        input_fn=batch_norm_elemt_input_fn,
        op_name="batch_norm_elemt",
        torch_op=torch_batch_norm_elemt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
