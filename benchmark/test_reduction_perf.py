import random
from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import BOOL_DTYPES, FLOAT_DTYPES, INT_DTYPES, BenchLevel
from benchmark.performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    SkipVersion,
    generate_tensor_input,
    unary_input_fn,
    vendor_name,
)
from flag_gems.utils import shape_utils


class UnaryReductionBenchmark(Benchmark):
    """
    Base class for benchmarking reduction operations.
    """

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        more_shapes_1d = [
            (1025 * 1024,),
            (1024 * 1024 * 1024,),
        ]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 21, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(0, 15, 4)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim > 1:
                yield inp, 1
            else:
                yield inp,


forward_operations = [
    ("all", torch.all, FLOAT_DTYPES),
    ("any", torch.any, FLOAT_DTYPES),
    ("amax", torch.amax, FLOAT_DTYPES),
    ("argmax", torch.argmax, FLOAT_DTYPES),
    ("argmin", torch.argmin, FLOAT_DTYPES),
    ("max", torch.max, FLOAT_DTYPES),
    ("mean", torch.mean, FLOAT_DTYPES),
    ("min", torch.min, FLOAT_DTYPES),
    ("prod", torch.prod, FLOAT_DTYPES),
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
    ("std", torch.std, FLOAT_DTYPES),
    ("sum", torch.sum, FLOAT_DTYPES),
    ("var_mean", torch.var_mean, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(name, op, dtype, marks=getattr(pytest.mark, name, None))
        for name, op, dtype in forward_operations
    ],
)
def test_general_reduction_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


backward_operations = [
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name, op, dtype, marks=getattr(pytest.mark, name + "_backward", None)
        )
        for name, op, dtype in backward_operations
    ],
)
def test_general_reduction_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()


def cross_entropy_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[-1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}
        yield inp, target, {
            "weight": weight,
            "reduction": "sum",
            "label_smoothing": 0.1,
        }


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[-1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


def cumsum_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, 1


def mse_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = generate_tensor_input(shape, cur_dtype, device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, target, {"reduction": "mean"}
        yield inp, target, {"reduction": "sum"}
        yield inp, target, {"reduction": "none"}


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "log_softmax",
            torch.nn.functional.log_softmax,
            unary_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.log_softmax,
        ),
        pytest.param(
            "nonzero",
            torch.nonzero,
            unary_input_fn,
            FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
            marks=pytest.mark.nonzero,
        ),
        pytest.param(
            "cross_entropy_loss",
            torch.nn.functional.cross_entropy,
            cross_entropy_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.cross_entropy_loss,
        ),
        pytest.param(
            "cumsum",
            torch.cumsum,
            cumsum_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.cumsum,
        ),
        pytest.param(
            "cummin",
            torch.cummin,
            cumsum_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.cummin,
        ),
        pytest.param(
            "cummax",
            torch.cummax,
            cumsum_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.cummax,
        ),
        pytest.param(
            "nll_loss",
            torch.nn.functional.nll_loss,
            nll_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.nll_loss,
        ),
        pytest.param(
            "mse_loss",
            torch.nn.functional.mse_loss,
            mse_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.mse_loss,
        ),
    ],
)
def test_generic_reduction_benchmark(op_name, torch_op, input_fn, dtypes):
    if vendor_name == "kunlunxin":
        if SkipVersion("torch", "<2.5"):
            if op_name in ["nll_loss"]:
                pytest.skip(
                    "INT16 is not supported in XPytorch 2.0. Please upgrade your PyTorch version >= 2.5"
                )
            if op_name in ["nonzero"]:
                pytest.skip(
                    "Not supported in XPytorch 2.0. Please upgrade your PyTorch version >= 2.5"
                )
    bench = GenericBenchmark2DOnly(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    if op_name == "cross_entropy_loss":
        bench.set_gems(flag_gems.cross_entropy_loss)
    bench.run()


# @pytest.mark.skipif(vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.count_nonzero
def test_perf_count_nonzero():
    def count_nonzero_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = random.choice([None, 0, 1])

        yield inp, dim

    bench = GenericBenchmark2DOnly(
        input_fn=count_nonzero_input_fn,
        op_name="count_nonzero",
        torch_op=torch.count_nonzero,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


def avg_pool2d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    # Common case
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # With count_include_pad=False
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": False,
            "divisor_override": None,
        }
        # With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": True,
            "count_include_pad": True,
            "divisor_override": None,
        }
        # With divisor_override
        if shape[-2] >= 2 and shape[-1] >= 2:
            yield inp, {
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": 3,
            }


class AvgPool2dBenchmark(GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_4d = [
            (4, 3, 224, 224),  # Typical input image size
            (16, 64, 56, 56),  # Early ResNet layer output
            (32, 128, 28, 28),  # Mid ResNet layer output
            (64, 256, 14, 14),  # Later ResNet layer output
            (128, 512, 7, 7),  # Final ResNet layer output
        ]

        for shape in shapes_4d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.avg_pool2d
def test_perf_avg_pool2d():
    bench = AvgPool2dBenchmark(
        input_fn=avg_pool2d_input_fn,
        op_name="avg_pool2d",
        torch_op=torch.ops.aten.avg_pool2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.avg_pool2d_backward
def test_perf_avg_pool2d_backward():
    bench = AvgPool2dBenchmark(
        input_fn=avg_pool2d_input_fn,
        op_name="avg_pool2d",
        torch_op=torch.ops.aten.avg_pool2d,
        dtypes=[torch.float32] if vendor_name == "mthreads" else FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()


def max_pool2d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # Non-square kernel/stride/padding
        if shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": (3, 5),
                "stride": (2, 1),
                "padding": (1, 2),
                "dilation": 1,
                "ceil_mode": False,
            }
        # With dilation
        yield inp, {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 2,
            "ceil_mode": False,
        }
        # With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": True,
        }


class MaxPool2dBenchmark(GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_4d = [
            (4, 3, 224, 224),  # Typical input image size
            (16, 64, 56, 56),  # Early ResNet layer output
            (32, 128, 28, 28),  # Mid ResNet layer output
            (64, 256, 14, 14),  # Later ResNet layer output
            (128, 512, 7, 7),  # Final ResNet layer output
        ]

        for shape in shapes_4d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.max_pool2d
def test_perf_max_pool2d():
    bench = MaxPool2dBenchmark(
        input_fn=max_pool2d_input_fn,
        op_name="max_pool2d",
        torch_op=torch.nn.functional.max_pool2d_with_indices,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.max_pool2d_with_indices)
    bench.run()


@pytest.mark.max_pool2d_backward
def test_perf_max_pool2d_backward():
    def max_pool2d_backward_input_fn(shape, dtype, device):
        for forward_args in max_pool2d_input_fn(shape, dtype, device):
            inp, params = forward_args
            inp.requires_grad_(True)
            output, indices = torch.nn.functional.max_pool2d_with_indices(inp, **params)
            grad_output = torch.randn_like(output)
            yield grad_output, inp, indices, params

    def torch_max_pool2d_backward_wrapper(grad_output, input, indices, **kwargs):
        output, _ = torch.nn.functional.max_pool2d_with_indices(input, **kwargs)
        grad_input = torch.autograd.grad(
            outputs=(output,), inputs=(input,), grad_outputs=(grad_output,)
        )
        return grad_input[0]

    bench = MaxPool2dBenchmark(
        input_fn=max_pool2d_backward_input_fn,
        op_name="max_pool2d_backward",
        torch_op=torch_max_pool2d_backward_wrapper,
        dtypes=FLOAT_DTYPES,
        is_backward=False,
    )

    bench.set_gems(flag_gems.max_pool2d_backward)
    bench.run()


@pytest.mark.dot
def test_perf_dot():
    def dot_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype=dtype, device=device)
        if inp.dim() > 1:
            inp = inp.flatten()
        yield inp, inp

    bench = GenericBenchmark(
        input_fn=dot_input_fn,
        op_name="dot",
        torch_op=torch.dot,
        dtypes=FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.trace
def test_perf_trace():
    def trace_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype=dtype, device=device)
        yield inp,

    bench = GenericBenchmark2DOnly(
        input_fn=trace_input_fn,
        op_name="trace",
        torch_op=torch.trace,
        dtypes=FLOAT_DTYPES if vendor_name == "mthreads" else FLOAT_DTYPES + INT_DTYPES,
    )

    bench.run()


class quantileBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        more_shapes_1d = [(4,), (1024,), (65535)]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 15, 3)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 3)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


def quantile_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    q = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=cur_dtype, device=device)
    yield inp, q, 0


@pytest.mark.skipif(True, reason="Skipping Triton version due to poor performance")
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "quantile",
            torch.quantile,
            quantile_input_fn,
            [torch.float32],
            marks=pytest.mark.quantile,
        )
    ],
)
def test_quantile_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = quantileBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


class ScaledSoftmaxBenchmark(GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        # shape: [batch, heads, query_len, key_len]
        shapes_small = [
            (1, 4, 64, 64),
            (2, 8, 128, 128),
            (4, 8, 256, 256),
        ]
        shapes_medium = [
            (8, 12, 512, 512),
            (16, 16, 1024, 1024),
            (32, 16, 512, 512),
        ]
        shapes_large = [
            (1, 32, 2048, 2048),
            (2, 40, 4096, 4096),
            # (4, 32, 8192, 8192),  # too big shape, out of memory
        ]
        shapes_4d = shapes_small + shapes_medium + shapes_large
        for shape in shapes_4d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.scaled_softmax
def test_perf_scaled_softmax_forward():
    try:
        from transformer_engine.common import _load_library

        _load_library()
        import transformer_engine_torch as tex  # type: ignore
    except ImportError:
        pytest.skip("TransformerEngine is not available, skipping performance test")

    def scaled_softmax_forward_input_fn(shape, dtype, device):
        S = generate_tensor_input(shape, dtype, device)
        scale_factor = 1 / S.shape[-1] ** 0.5
        yield S, scale_factor

    bench = ScaledSoftmaxBenchmark(
        input_fn=scaled_softmax_forward_input_fn,
        op_name="scaled_softmax_forward",
        torch_op=tex.scaled_softmax_forward,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(flag_gems.scaled_softmax_forward)
    bench.run()


@pytest.mark.scaled_softmax
def test_perf_scaled_softmax_backward():
    try:
        from transformer_engine.common import _load_library

        _load_library()
        import transformer_engine_torch as tex  # type: ignore
    except ImportError:
        pytest.skip("TransformerEngine is not available, skipping performance test")

    def scaled_softmax_backward_input_fn(shape, dtype, device):
        S = generate_tensor_input(shape, dtype, device)
        scale_factor = 1 / S.shape[-1] ** 0.5
        P = torch.softmax(S / scale_factor, dim=-1)
        dP = generate_tensor_input(shape, dtype, device)
        yield P, dP, scale_factor

    bench = ScaledSoftmaxBenchmark(
        input_fn=scaled_softmax_backward_input_fn,
        op_name="scaled_softmax_backward",
        torch_op=tex.scaled_softmax_backward,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(flag_gems.scaled_softmax_backward)
    bench.run()
