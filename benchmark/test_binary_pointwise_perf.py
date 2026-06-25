from typing import Generator

import pytest
import torch

from .attri_util import BOOL_DTYPES, DEFAULT_METRICS, FLOAT_DTYPES, INT_DTYPES
from .performance_utils import Benchmark, generate_tensor_input


def _bench_marks(name, doc_mark=None):
    marks = getattr(pytest.mark, name)
    if doc_mark:
        return [marks, getattr(pytest.mark, doc_mark)]
    return marks


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
            marks=_bench_marks(name, doc_mark),
        )
        for name, op, dtype, doc_mark in [
            # Arithmetic operations
            ("add", torch.add, FLOAT_DTYPES, "add_tensor"),
            ("div", torch.div, FLOAT_DTYPES, "true_divide"),
            ("mul", torch.mul, FLOAT_DTYPES, None),
            ("pow", torch.pow, FLOAT_DTYPES, "pow_tensor_tensor"),
            ("sub", torch.sub, FLOAT_DTYPES, None),
            ("floor_divide", torch.floor_divide, INT_DTYPES, None),
            ("remainder", torch.remainder, INT_DTYPES, None),
            ("rsub", torch.rsub, FLOAT_DTYPES, None),
            ("logical_or", torch.logical_or, INT_DTYPES + BOOL_DTYPES, None),
            ("logical_and", torch.logical_and, INT_DTYPES + BOOL_DTYPES, None),
            ("logical_xor", torch.logical_xor, INT_DTYPES + BOOL_DTYPES, None),
            # Comparison operations
            ("eq", torch.eq, FLOAT_DTYPES, None),
            ("ge", torch.ge, FLOAT_DTYPES, None),
            ("gt", torch.gt, FLOAT_DTYPES, None),
            ("le", torch.le, FLOAT_DTYPES, None),
            ("lt", torch.lt, FLOAT_DTYPES, None),
            ("ne", torch.ne, FLOAT_DTYPES, None),
            # Minimum and maximum operations
            ("maximum", torch.maximum, FLOAT_DTYPES, None),
            ("minimum", torch.minimum, FLOAT_DTYPES, None),
            # Bitwise operations
            ("bitwise_and", torch.bitwise_and, INT_DTYPES + BOOL_DTYPES, "bitwise_and_tensor"),
            ("bitwise_or", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES, "bitwise_or_tensor"),
            ("or_", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES, "bitwise_or_tensor"),
            # Numerical Checks
            ("isclose", torch.isclose, FLOAT_DTYPES + INT_DTYPES, None),
            ("allclose", torch.allclose, FLOAT_DTYPES + INT_DTYPES, None),
        ]
    ],
)
def test_general_binary_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()
