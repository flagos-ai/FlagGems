from typing import Generator

import pytest
import torch

from benchmark.attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from benchmark.performance_utils import Benchmark, generate_tensor_input


class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)
        shape2 = list(args[0].shape)
        return torch.tensor(shape1).prod().item() + torch.tensor(shape2).prod().item()


class ScalarBinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations with scalar input.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = 0.001  # Scalar input
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[1].shape)  # Second argument is the tensor
        return torch.tensor(shape).prod().item()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in [
            ("add", torch.add, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("allclose", torch.allclose, FLOAT_DTYPES + INT_DTYPES),
            ("bitwise_and", torch.bitwise_and, INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES),
            ("div", torch.div, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("gcd", torch.gcd, INT_DTYPES),
            ("dunder_or", lambda a, b: a | b, INT_DTYPES + BOOL_DTYPES),
            ("eq", torch.eq, FLOAT_DTYPES),
            ("equal", torch.equal, FLOAT_DTYPES),
            ("floor_divide", torch.floor_divide, INT_DTYPES),
            ("fmin", torch.fmin, FLOAT_DTYPES),
            ("gcd", torch.gcd, INT_DTYPES),
            ("ge", torch.ge, FLOAT_DTYPES),
            ("greater", torch.greater, FLOAT_DTYPES),
            ("gt", torch.gt, FLOAT_DTYPES),
            ("hypot", torch.hypot, FLOAT_DTYPES),
            ("isclose", torch.isclose, FLOAT_DTYPES + INT_DTYPES),
            ("le", torch.le, FLOAT_DTYPES),
            ("logaddexp", torch.logaddexp, FLOAT_DTYPES),
            ("logical_and", torch.logical_and, INT_DTYPES + BOOL_DTYPES),
            ("logical_or", torch.logical_or, INT_DTYPES + BOOL_DTYPES),
            ("logical_xor", torch.logical_xor, INT_DTYPES + BOOL_DTYPES),
            ("lt", torch.lt, FLOAT_DTYPES),
            ("maximum", torch.maximum, FLOAT_DTYPES),
            ("minimum", torch.minimum, FLOAT_DTYPES),
            ("mul", torch.mul, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("ne", torch.ne, FLOAT_DTYPES),
            ("polar", torch.polar, [torch.float32]),
            ("pow", torch.pow, FLOAT_DTYPES),
            ("remainder", torch.remainder, INT_DTYPES),
            ("sub", torch.sub, FLOAT_DTYPES + COMPLEX_DTYPES),
        ]
    ],
)
def test_general_binary_pointwise(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in [
            ("add_", lambda a, b: a.add_(b), FLOAT_DTYPES),
            ("bitwise_and_", lambda a, b: a.bitwise_and_(b), INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or_", lambda a, b: a.bitwise_or_(b), INT_DTYPES + BOOL_DTYPES),
            ("div_", lambda a, b: a.div_(b), FLOAT_DTYPES),
            ("dunder_ior", lambda a, b: a.__ior__(b), INT_DTYPES + BOOL_DTYPES),
            ("floor_divide_", lambda a, b: a.floor_divide_(b), INT_DTYPES),
            ("logical_and_", lambda a, b: a.logical_and_(b), INT_DTYPES + BOOL_DTYPES),
            ("logical_or_", lambda a, b: a.logical_or_(b), INT_DTYPES + BOOL_DTYPES),
            ("mul_", lambda a, b: a.mul_(b), FLOAT_DTYPES),
            ("pow_", lambda a, b: a.pow_(b), FLOAT_DTYPES),
            ("remainder_", lambda a, b: a.remainder_(b), INT_DTYPES),
            ("sub_", lambda a, b: a.sub_(b), FLOAT_DTYPES),
        ]
    ],
)
def test_general_inplace_binary_pointwise(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=dtypes, is_inplace=True
    )
    bench.run()


@pytest.mark.pow
def test_pow(op_name, torch_op, dtypes):
    bench = ScalarBinaryPointwiseBenchmark(
        op_name="pow",
        torch_op=lambda a, b: torch.pow(a, b),
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
