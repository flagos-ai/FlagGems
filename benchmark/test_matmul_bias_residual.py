import pytest
import torch

import flag_gems

from . import base, consts


def matmul_bias_residual_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp = torch.randn([m, k], dtype=cur_dtype, device=device)
    weight = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([n], dtype=cur_dtype, device=device)
    residual = torch.randn([m, n], dtype=cur_dtype, device=device)
    yield inp, weight, bias, residual


def matmul_bias_residual_ref(inp, weight, bias, residual, alpha=1.0, beta=1.0):
    """Reference implementation using torch operations."""
    out = torch.addmm(bias, inp, weight, alpha=alpha, beta=beta)
    out = out + residual
    return out


class MatmulBiasResidualBenchmark(base.BlasBenchmark):
    def get_tflops(self, op, *args, **kwargs):
        # Matmul-bias-residual computes MxN dot products plus bias and residual adds.
        return args[0].shape[0] * args[1].shape[1] * (args[0].shape[1] * 2 + 2)


@pytest.mark.matmul_bias_residual
def test_matmul_bias_residual():
    bench = MatmulBiasResidualBenchmark(
        input_fn=matmul_bias_residual_input_fn,
        op_name="matmul_bias_residual",
        torch_op=matmul_bias_residual_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.matmul_bias_residual)
    bench.run()
