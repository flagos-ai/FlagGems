import os

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES, BenchLevel
from benchmark.performance_utils import (
    Config,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    GenericBenchmarkExcluse1D,
    SkipVersion,
    unary_input_fn,
    vendor_name,
)


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


def instancenorm_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = None
    running_var = None
    use_input_stats = True
    momentum = 0.1
    eps = 1e-5
    cudnn_enabled = True
    yield inp, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        running_mean = torch.randn((C,), dtype=dtype, device=device)
        running_var = torch.randn((C,), dtype=dtype, device=device)
        yield inp, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled


def batchnorm_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = None
    running_var = None
    training = True
    momentum = 0.1
    eps = 1e-5
    cudnn_enabled = True
    yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        running_mean = torch.randn((C,), dtype=dtype, device=device)
        running_var = torch.randn((C,), dtype=dtype, device=device)
        yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled


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
        pytest.param(
            "instance_norm",
            torch.instance_norm,
            instancenorm_input_fn,
            marks=pytest.mark.instance_norm,
        ),
        pytest.param(
            "batch_norm",
            torch.batch_norm,
            batchnorm_input_fn,
            marks=pytest.mark.batch_norm,
        ),
    ],
)
def test_group_and_layer_and_instance_norm_benchmark(op_name, torch_op, input_fn):
    if vendor_name == "kunlunxin" and op_name in [
        "instance_norm",
        "batch_norm",
    ]:
        pytest.skip("RUNTIME TODOFIX.(batch_norm unsupported in torch)")
    if vendor_name == "mthreads" and op_name == "instance_norm":
        # Compatible with older versions of LLVM
        os.environ["DISABLE_LLVM_OPT"] = "1"
    bench = NormBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )
    if op_name == "instance_norm":
        bench.set_gems(flag_gems.instance_norm)
    bench.run()
    if vendor_name == "mthreads" and op_name == "instance_norm":
        del os.environ["DISABLE_LLVM_OPT"]


def weight_norm_interface_input_fn(shape, dtype, device):
    dim = 0
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(shape[dim], dtype=dtype, device=device)
    yield v, g, dim


def weight_norm_input_fn(shape, dtype, device):
    v = torch.randn(shape, dtype=dtype, device=device)
    if vendor_name == "cambricon":
        # Cambricon fix input shape limit.
        g = torch.randn(shape[:1] + (1,) * (len(shape) - 1), dtype=dtype, device=device)
    else:
        g = torch.randn(shape, dtype=dtype, device=device)
    yield v, g, 0


norm_operations = [
    ("weight_norm", torch._weight_norm, weight_norm_input_fn),
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
    bench = GenericBenchmarkExcluse1D(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op
    )
    if op_name == "weight_norm":
        bench.set_gems(flag_gems.weight_norm)
    bench.run()


@pytest.mark.rms_norm
@pytest.mark.skipif(
    SkipVersion("torch", "<2.4"),
    reason="The version prior to 2.4 does not include the rms_norm API in torch.",
)
def test_perf_rms_norm():
    def rms_norm_input_fn(shape, dtype, device):
        M, N = shape
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield inp, (N,), weight

    bench = GenericBenchmark2DOnly(
        input_fn=rms_norm_input_fn,
        op_name="rms_norm",
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.run()
