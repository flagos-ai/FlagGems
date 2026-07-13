import pytest
import torch

import flag_gems

from . import base, consts


class MatmulBiasActivationBenchmark(base.BlasBenchmark):
    """
    benchmark for matmul_bias_activation
    """

    def set_more_shapes(self):
        return None


def matmul_bias_activation_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    # Note: b is ignored as we use (m, k) x (k, n) + bias
    input_tensor = torch.randn([m, k], dtype=cur_dtype, device=device)
    weight = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([n], dtype=cur_dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.matmul_bias_activation
def test_matmul_bias_activation():
    def mma_torch_op(input, weight, bias):
        return torch.relu(torch.mm(input, weight) + bias)

    bench = MatmulBiasActivationBenchmark(
        input_fn=matmul_bias_activation_input_fn,
        op_name="matmul_bias_activation",
        torch_op=mma_torch_op,
        gems_op=flag_gems.matmul_bias_activation,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
