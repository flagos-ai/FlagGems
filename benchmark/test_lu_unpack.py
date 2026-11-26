import pytest
import torch

import flag_gems

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


class LuUnpackBackwardBenchmark(base.GenericBenchmark2DOnly):
    """Backward of `lu_unpack`: repack grad_L/grad_U into the packed grad_LU.

    The forward runs once per case outside the timed closure; the timer only
    calls `torch.autograd.grad` on the two differentiable unpacked matrices, so
    the measured latency is the packed-LU gradient path (the ATen composite
    `tril`/`triu`/`add`/`cat` natively, the fused FlagGems kernel under gems
    dispatch).
    """

    def _benchmark_callable(self, op, *args, **kwargs):
        LU_data, LU_pivots = args[:2]
        _, L, U = op(LU_data, LU_pivots)
        grad_L = torch.randn_like(L)
        grad_U = torch.randn_like(U)
        fn = lambda: torch.autograd.grad(
            (L, U), (LU_data,), (grad_L, grad_U), retain_graph=True
        )
        return fn, [LU_data]


@pytest.mark.lu_unpack_backward
def test_lu_unpack_backward():
    def lu_unpack_backward_input_fn(shape, dtype, device):
        # Only 2D core shapes are used; the packed LU data comes from the same
        # native factorization the forward benchmark uses.
        if len(shape) != 2:
            return
        m, n = shape
        A = torch.randn(m, n, dtype=dtype, device=device)
        LU, pivots = torch.linalg.lu_factor(A)
        yield LU, pivots

    bench = LuUnpackBackwardBenchmark(
        input_fn=lu_unpack_backward_input_fn,
        op_name="lu_unpack",
        torch_op=torch.ops.aten.lu_unpack,
        # torch.linalg.lu_factor only supports float32/float64
        dtypes=[torch.float32, torch.float64],
        is_backward=True,
    )
    bench.set_gems(flag_gems.lu_unpack)
    bench.run()
