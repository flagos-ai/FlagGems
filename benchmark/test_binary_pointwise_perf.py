from typing import Generator

import pytest
import torch

from benchmark.attri_util import BOOL_DTYPES, DEFAULT_METRICS, FLOAT_DTYPES, INT_DTYPES
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
            # Arithmetic operations
            ("add", torch.add, FLOAT_DTYPES),
            ("div", torch.div, FLOAT_DTYPES),
            ("mul", torch.mul, FLOAT_DTYPES),
            ("sub", torch.sub, FLOAT_DTYPES),
            ("pow", torch.pow, FLOAT_DTYPES),
            ("polar", torch.polar, [torch.float32]),
            ("floor_divide", torch.floor_divide, INT_DTYPES),
            ("remainder", torch.remainder, INT_DTYPES),
            ("logical_or", torch.logical_or, INT_DTYPES + BOOL_DTYPES),
            ("logical_and", torch.logical_and, INT_DTYPES + BOOL_DTYPES),
            ("logical_xor", torch.logical_xor, INT_DTYPES + BOOL_DTYPES),
            # Comparison operations
            ("eq", torch.eq, FLOAT_DTYPES),
            ("equal", torch.equal, FLOAT_DTYPES),
            ("ge", torch.ge, FLOAT_DTYPES),
            ("gt", torch.gt, FLOAT_DTYPES),
            ("le", torch.le, FLOAT_DTYPES),
            ("lt", torch.lt, FLOAT_DTYPES),
            ("ne", torch.ne, FLOAT_DTYPES),
            # Minimum and maximum operations
            ("maximum", torch.maximum, FLOAT_DTYPES),
            ("minimum", torch.minimum, FLOAT_DTYPES),
            # Bitwise operations
            ("bitwise_and", torch.bitwise_and, INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES),
            # Numerical Checks
            ("isclose", torch.isclose, FLOAT_DTYPES + INT_DTYPES),
            ("allclose", torch.allclose, FLOAT_DTYPES + INT_DTYPES),
        ]
    ],
)
def test_general_binary_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


class BinaryScalarPointwiseBenchmark(Benchmark):
    """
    Benchmark class for binary pointwise operations with a scalar operand.
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
        return torch.tensor(shape1).prod().item()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            "ne_scalar",
            torch.ne,
            FLOAT_DTYPES,
            marks=pytest.mark.ne,
        ),
        pytest.param(
            "lt_scalar",
            torch.lt,
            FLOAT_DTYPES,
            marks=pytest.mark.lt,
        ),
        pytest.param(
            "ge_scalar",
            torch.ge,
            FLOAT_DTYPES,
            marks=pytest.mark.ge,
        ),
    ],
)
def test_binary_scalar_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryScalarPointwiseBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


class BinaryTensorScalarBenchmark(Benchmark):
    """
    Benchmark class for binary pointwise operations with tensor and scalar operands
    (e.g., pow_tensor_scalar).
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, scalar=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar = scalar

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, self.scalar

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)
        return torch.tensor(shape1).prod().item()


@pytest.mark.pow
def test_pow_tensor_scalar_perf():
    bench = BinaryTensorScalarBenchmark(
        op_name="pow_tensor_scalar",
        torch_op=torch.pow,
        dtypes=FLOAT_DTYPES,
        scalar=2.0,
    )
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
            # Arithmetic operations
            ("add_", lambda a, b: a.add_(b), FLOAT_DTYPES),
            ("div_", lambda a, b: a.div_(b), FLOAT_DTYPES),
            ("mul_", lambda a, b: a.mul_(b), FLOAT_DTYPES),
            ("sub_", lambda a, b: a.sub_(b), FLOAT_DTYPES),
            ("pow_", lambda a, b: a.pow_(b), FLOAT_DTYPES),
            ("floor_divide_", lambda a, b: a.floor_divide_(b), INT_DTYPES),
            ("remainder_", lambda a, b: a.remainder_(b), INT_DTYPES),
            ("logical_or_", lambda a, b: a.logical_or_(b), INT_DTYPES + BOOL_DTYPES),
            ("logical_and_", lambda a, b: a.logical_and_(b), INT_DTYPES + BOOL_DTYPES),
            # Bitwise operations
            ("bitwise_and_", lambda a, b: a.bitwise_and_(b), INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or_", lambda a, b: a.bitwise_or_(b), INT_DTYPES + BOOL_DTYPES),
        ]
    ],
)
def test_general_inplace_binary_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=dtypes, is_inplace=True
    )
    bench.run()
