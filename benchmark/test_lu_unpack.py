import pytest
import torch

from . import base


@pytest.mark.lu_unpack
def test_lu_unpack():
    def lu_unpack_input_fn(shape, dtype, device):
        # shape here is a tuple from the auto-generated shapes
        # We only use 2D shapes for lu_unpack
        if len(shape) != 2:
            return
        m, n = shape
        # Generate a random matrix and compute LU factorization
        # We precompute LU and pivots to focus on the lu_unpack performance
        A = torch.randn(m, n, dtype=dtype, device=device)
        LU, pivots = torch.linalg.lu_factor(A)
        yield LU, pivots

    bench = base.GenericBenchmark2DOnly(
        input_fn=lu_unpack_input_fn,
        op_name="lu_unpack",
        torch_op=torch.ops.aten.lu_unpack,
        # torch.linalg.lu_factor only supports float32/float64; half/bfloat16 not supported
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
