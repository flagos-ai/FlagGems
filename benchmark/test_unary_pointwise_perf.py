from typing import Generator

import pytest
import torch

from .attri_util import BOOL_DTYPES, DEFAULT_METRICS, FLOAT_DTYPES, INT_DTYPES
from .performance_utils import Benchmark, GenericBenchmark, generate_tensor_input


class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, input_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 1), (1024, 16), (1024, 256), (1024, 512)]
        sp_shapes_3d = [(64, 64, 1), (64, 64, 16), (64, 64, 32)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            if self.input_fn is not None:
                yield from self.input_fn(shape, cur_dtype, self.device)
            else:
                inp = generate_tensor_input(shape, cur_dtype, self.device)
                yield inp,

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


forward_operations = [
    ("abs", torch.abs, FLOAT_DTYPES),
    ("acos", torch.acos, FLOAT_DTYPES),
    ("alias_copy", torch.ops.aten.alias_copy, FLOAT_DTYPES),
    ("arcsinh", torch.arcsinh, FLOAT_DTYPES),
    ("ceil", torch.ceil, FLOAT_DTYPES),
    ("erf", torch.erf, FLOAT_DTYPES),
    ("exp", torch.exp, FLOAT_DTYPES),
    ("i0", torch.i0, FLOAT_DTYPES),
    ("neg", torch.neg, FLOAT_DTYPES),
    ("reciprocal", torch.reciprocal, FLOAT_DTYPES),
    ("rsqrt", torch.rsqrt, FLOAT_DTYPES),
    ("special_i0e", torch.ops.aten.special_i0e, FLOAT_DTYPES),
    ("special_i1", torch.special.i1, FLOAT_DTYPES),
    ("logical_not", torch.logical_not, INT_DTYPES + BOOL_DTYPES),
    # ("triu", torch.triu, FLOAT_DTYPES),  # do not support 1d shapes
    # Dropout
    ("native_dropout", torch.nn.Dropout(p=0.5), FLOAT_DTYPES),
    ("dropout", torch.nn.Dropout(p=0.5), FLOAT_DTYPES),
    # Activation operations
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
    ("hardsigmoid", torch.nn.functional.hardsigmoid, FLOAT_DTYPES),
    ("relu", torch.nn.functional.relu, FLOAT_DTYPES),
    ("relu6", torch.nn.functional.relu6, FLOAT_DTYPES),
    ("selu", torch.nn.functional.selu, FLOAT_DTYPES),
    ("sigmoid", torch.sigmoid, FLOAT_DTYPES),
    ("silu", torch.nn.functional.silu, FLOAT_DTYPES),
    ("softshrink", torch.nn.functional.softshrink, FLOAT_DTYPES),
    # Trigonometric operations
    ("cos", torch.cos, FLOAT_DTYPES),
    ("sin", torch.sin, FLOAT_DTYPES),
    ("tanh", torch.tanh, FLOAT_DTYPES),
    # Bitwise operations
    ("bitwise_not", torch.bitwise_not, INT_DTYPES),
    # Numerical Checks
    ("isinf", torch.isinf, FLOAT_DTYPES),
    ("isnan", torch.isnan, FLOAT_DTYPES),
    ("isfinite", torch.isfinite, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in forward_operations
    ],
)
def test_general_unary_pointwise_perf(op_name, torch_op, dtypes):
    bench = UnaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


def _clone_then(torch_op):
    def wrapped(inp):
        return torch_op(inp.clone())

    return wrapped


def arctanh_input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) * 1.8 - 0.9
    yield inp,


def log1p_input_fn(shape, dtype, device):
    yield torch.rand(shape, dtype=dtype, device=device),


def logit_input_fn(shape, dtype, device):
    base = torch.empty(shape, device=device, dtype=torch.float32).uniform_(-4.0, 4.0)
    yield torch.sigmoid(base).to(dtype=dtype),


custom_unary_operations = [
    ("arctanh_", _clone_then(torch.Tensor.arctanh_), FLOAT_DTYPES, arctanh_input_fn),
    ("floor_", _clone_then(torch.Tensor.floor_), FLOAT_DTYPES, None),
    ("log1p_", _clone_then(torch.Tensor.log1p_), FLOAT_DTYPES, log1p_input_fn),
    ("logit", lambda inp: torch.logit(inp, eps=1e-6), FLOAT_DTYPES, logit_input_fn),
    ("sgn_", _clone_then(torch.Tensor.sgn_), FLOAT_DTYPES, None),
    ("sinh_", _clone_then(torch.Tensor.sinh_), FLOAT_DTYPES, None),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes, input_fn",
    [
        pytest.param(
            name,
            op,
            dtype,
            input_fn,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype, input_fn in custom_unary_operations
    ],
)
def test_custom_unary_pointwise_perf(op_name, torch_op, dtypes, input_fn):
    bench = UnaryPointwiseBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        input_fn=input_fn,
    )
    bench.run()


@pytest.mark.gelu_backward
def test_perf_gelu_backward():
    bench = UnaryPointwiseBenchmark(
        op_name="gelu_backward",
        torch_op=torch.nn.functional.gelu,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()


@pytest.mark.elu_backward
def test_perf_elu_backward():
    def elu_backward_input_fn(shape, dtype, device):
        alpha = torch.rand(1).item()
        scale = 1.0
        input_scale = 1.0
        inp = torch.randn(shape, dtype=dtype, device=device)
        grad_out = torch.randn_like(inp)
        result = torch.ops.aten.elu(inp, alpha, scale, input_scale)

        yield grad_out, alpha, scale, input_scale, True, result
        yield grad_out, alpha, scale, input_scale, False, inp

    bench = GenericBenchmark(
        input_fn=elu_backward_input_fn,
        op_name="elu_backward",
        torch_op=torch.ops.aten.elu_backward,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.prelu
def test_perf_prelu():
    class PReLUBenchmark(GenericBenchmark):
        DEFAULT_SHAPES = [
            (2, 3),
            (128, 256),
            (512, 512),
            (4, 8, 16),
            (2, 32, 16, 4),
            (2, 128, 16, 4),
        ]
        DEFAULT_SHAPE_DESC = "N, C, *"

        def set_shapes(self, shape_file_path=None):
            self.shapes = [tuple(shape) for shape in self.DEFAULT_SHAPES]
            self.shape_desc = self.DEFAULT_SHAPE_DESC

        def set_more_shapes(self):
            return None

    def prelu_input_fn(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=device)
        yield x, torch.randn((), dtype=dtype, device=device)
        yield x, torch.randn((shape[1],), dtype=dtype, device=device)

    bench = PReLUBenchmark(
        input_fn=prelu_input_fn,
        op_name="prelu",
        torch_op=torch.ops.aten.prelu,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.rrelu_with_noise_backward
def test_perf_rrelu_with_noise_backward():
    def rrelu_with_noise_backward_input_fn(shape, dtype, device):
        grad_output = torch.randn(shape, dtype=dtype, device=device)
        inp = torch.randn(shape, dtype=dtype, device=device)
        noise = torch.rand(shape, dtype=dtype, device=device)
        yield grad_output, inp, noise, 0.125, 1.0 / 3.0, True, False

    bench = GenericBenchmark(
        input_fn=rrelu_with_noise_backward_input_fn,
        op_name="rrelu_with_noise_backward",
        torch_op=torch.ops.aten.rrelu_with_noise_backward,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
