import pytest
import torch

import flag_gems

from . import base, consts


def matmuladd_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn(
        [
            n,
        ],
        dtype=cur_dtype,
        device=device,
    )
    yield inp1, inp2, bias


def matmuladd_torch_wrapper(inp1, inp2, bias):
    """Wrapper to use torch.matmul + bias as reference for MatMulAdd"""
    return torch.matmul(inp1, inp2) + bias


class MatMulAddBenchmark(base.BlasBenchmark):
    def get_tflops(self, op, *args, **kwargs):
        # MatMulAdd computes MxN dot products plus one bias add per output element.
        return args[0].shape[0] * args[1].shape[1] * (args[0].shape[1] * 2 + 1)


@pytest.mark.matmuladd
def test_matmuladd():
    bench = MatMulAddBenchmark(
        input_fn=matmuladd_input_fn,
        op_name="matmuladd",
        torch_op=matmuladd_torch_wrapper,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.matmuladd)
    bench.run()
