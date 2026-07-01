import pytest
import torch

import flag_gems

from . import base, consts


def matmul_layernorm_input_fn(m, n, cur_dtype, device):
    input_tensor = torch.randn(m, n, dtype=cur_dtype, device=device)
    weight = torch.randn(n, n, dtype=cur_dtype, device=device)
    bias = torch.randn(n, dtype=cur_dtype, device=device)
    yield input_tensor, weight, bias


class MatmulLayerNormBenchmark(base.GenericBenchmark2DOnly):
    """
    Benchmark for matmul_layernorm operation
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for m, n in self.shapes:
            yield from matmul_layernorm_input_fn(m, n, cur_dtype, self.device)


def matmul_layernorm_torch_op(input, weight, bias):
    matmul_result = torch.matmul(input, weight.t()) + bias
    return torch.nn.functional.layer_norm(
        matmul_result, [matmul_result.shape[-1]], eps=1e-5
    )


@pytest.mark.matmul_layernorm
def test_matmul_layernorm():
    bench = MatmulLayerNormBenchmark(
        input_fn=matmul_layernorm_input_fn,
        op_name="matmul_layernorm",
        torch_op=matmul_layernorm_torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.matmul_layernorm)
    bench.run()
