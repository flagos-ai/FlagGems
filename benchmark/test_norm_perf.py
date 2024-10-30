import pytest
import torch

from .attri_util import FLOAT_DTYPES, BenchLevel
from .conftest import Config
from .performance_utils import GenericBenchmark, unary_input_fn


class NormBenchmark(GenericBenchmark):
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


def groupnorm_input_fn(shape, dtype, device):
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
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, channel, weight, bias


def layernorm_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = shape[1:]
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    bias = torch.randn(layer_shape, dtype=dtype, device=device)
    yield inp, layer_shape, weight, bias


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "group_norm",
            torch.nn.functional.group_norm,
            groupnorm_input_fn,
            marks=pytest.mark.group_norm,
        ),
        pytest.param(
            "layer_norm",
            torch.layer_norm,
            layernorm_input_fn,
            marks=pytest.mark.layer_norm,
        ),
    ],
)
def test_group_and_layer_norm_benchmark(op_name, torch_op, input_fn):
    bench = NormBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )
    bench.run()


def weight_norm_input_fn(shape, dtype, device):
    dim = 0
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(shape[dim], dtype=dtype, device=device)
    yield v, g, dim


norm_operations = [
    ("weight_norm_interface", torch._weight_norm_interface, weight_norm_input_fn),
    ("vector_norm", torch.linalg.vector_norm, unary_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in norm_operations
    ],
)
def test_weight_vector_norm_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(input_fn=input_fn, op_name=op_name, torch_op=torch_op)
    bench.run()
