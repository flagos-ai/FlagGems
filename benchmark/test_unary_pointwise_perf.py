from typing import Generator

import pytest
import torch

from .attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from .performance_utils import Benchmark, generate_tensor_input, vendor_name


class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


forward_operations = [
    ("abs", torch.abs, FLOAT_DTYPES),
    (
        "angle",
        torch.angle,
        COMPLEX_DTYPES + [torch.float32] + INT_DTYPES + BOOL_DTYPES,
    ),
    ("erf", torch.erf, FLOAT_DTYPES),
    ("exp", torch.exp, FLOAT_DTYPES),
    ("neg", torch.neg, FLOAT_DTYPES),
    ("reciprocal", torch.reciprocal, FLOAT_DTYPES),
    ("rsqrt", torch.rsqrt, FLOAT_DTYPES),
    ("logical_not", torch.logical_not, INT_DTYPES + BOOL_DTYPES),
    ("log", torch.log, FLOAT_DTYPES),
    # ("triu", torch.triu, FLOAT_DTYPES),  # do not support 1d shapes
    # Dropout
    ("native_dropout", torch.nn.Dropout(p=0.5), FLOAT_DTYPES),
    ("dropout", torch.nn.Dropout(p=0.5), FLOAT_DTYPES),
    # Activation operations
    ("elu", torch.nn.functional.elu, FLOAT_DTYPES),
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
    ("relu", torch.nn.functional.relu, FLOAT_DTYPES),
    ("sigmoid", torch.sigmoid, FLOAT_DTYPES),
    ("log_sigmoid", torch.nn.functional.logsigmoid, FLOAT_DTYPES),
    ("silu", torch.nn.functional.silu, FLOAT_DTYPES),
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
    if vendor_name == "kunlunxin" and op_name == "elu":
        pytest.skip("RUNTIME TODOFIX")
    bench = UnaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


backward_operations = [
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name + "_backward", None),
        )
        for name, op, dtype in backward_operations
    ],
)
def test_general_unary_pointwise_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryPointwiseBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()
