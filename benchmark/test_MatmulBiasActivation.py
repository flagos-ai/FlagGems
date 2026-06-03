import pytest
import torch

import flag_gems

from . import base, consts


class MatmulBiasActivationBenchmark(base.BlasBenchmark):
    """
    benchmark for MatmulBiasActivation
    """

    def set_more_shapes(self):
        return None


def MatmulBiasActivation_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    # Note: b is ignored as we use (m, k) x (k, n) + bias
    input_tensor = torch.randn([m, k], dtype=cur_dtype, device=device)
    weight = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([n], dtype=cur_dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.MatmulBiasActivation
def test_MatmulBiasActivation():
    def mma_torch_op(input, weight, bias):
        return torch.relu(torch.mm(input, weight) + bias)

    bench = MatmulBiasActivationBenchmark(
        input_fn=MatmulBiasActivation_input_fn,
        op_name="MatmulBiasActivation",
        torch_op=mma_torch_op,
        gems_op=flag_gems.MatmulBiasActivation,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
