from typing import Generator

import pytest
import torch

from . import base, consts

# Shapes cover 1D, 2D, and 3D tensor products for trilinear operations
DEFAULT_SHAPES = [
    (1024,),
    (64, 64),
    (256, 256),
]


class TrilinearBenchmark(base.GenericBenchmark):
    # Benchmark shapes chosen to test various tensor dimensions for trilinear product
    DEFAULT_SHAPES = [
        (1024,),
        (64, 64),
        (256, 256),
    ]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)


def _input_fn(shape, cur_dtype, device):
    i1 = torch.randn(shape, dtype=cur_dtype, device=device)
    i2 = torch.randn(shape, dtype=cur_dtype, device=device)
    i3 = torch.randn(shape, dtype=cur_dtype, device=device)
    expand1 = []
    expand2 = []
    expand3 = []
    sumdim = []
    # unroll_dim must be in range [0, ndim-1]
    unroll_dim = 0

    yield i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim


@pytest.mark.trilinear
def test_trilinear():
    bench = TrilinearBenchmark(
        op_name="trilinear",
        input_fn=_input_fn,
        torch_op=torch._trilinear,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _input_fn_out(shape, cur_dtype, device):
    i1 = torch.randn(shape, dtype=cur_dtype, device=device)
    i2 = torch.randn(shape, dtype=cur_dtype, device=device)
    i3 = torch.randn(shape, dtype=cur_dtype, device=device)
    out = torch.empty(shape, dtype=cur_dtype, device=device)
    expand1 = []
    expand2 = []
    expand3 = []
    sumdim = []
    # unroll_dim must be in range [0, ndim-1]
    unroll_dim = 0

    yield i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim, out


@pytest.mark.underscore_trilinear_out
def test_trilinear_out():
    bench = TrilinearBenchmark(
        op_name="_trilinear_out",
        input_fn=_input_fn_out,
        torch_op=lambda i1, i2, i3, e1, e2, e3, sd, ud, out: torch.ops.aten._trilinear.out(
            i1, i2, i3, e1, e2, e3, sd, ud, out=out
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
