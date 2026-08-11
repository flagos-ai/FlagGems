import pytest
import torch

import flag_gems
from flag_gems.ops.awq_gemm import pack_awq_weight

from . import base

# (M, N, K) with K % GROUP_SIZE == 0 and N % 8 == 0
AWQ_SHAPES = [
    (1, 4096, 11008),  # decode
    (1, 11008, 4096),
    (16, 4096, 11008),
    (64, 4096, 11008),
    (256, 4096, 11008),
    (2048, 4096, 11008),
    (64, 11008, 4096),
]
GROUP_SIZE = 128


def torch_awq_gemm(x, qweight, qzeros, scales, group_size):
    """Naive baseline: dequantize the packed INT4 weight then torch matmul."""
    K = x.shape[1]
    N = qweight.shape[1]
    G = K // group_size
    w_code = torch.zeros(K, N, dtype=torch.int32, device=x.device)
    for i in range(8):
        w_code[i::8, :] = (qweight >> (4 * i)) & 0xF
    z_n = torch.zeros(G, N, dtype=torch.int32, device=x.device)
    for j in range(8):
        z_n[:, j::8] = (qzeros >> (4 * j)) & 0xF
    w = w_code - z_n.repeat_interleave(group_size, dim=0)
    w = w.to(x.dtype) * scales.repeat_interleave(group_size, dim=0)
    return x @ w


class AWQGemmBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        self.shapes = AWQ_SHAPES
        return []


@pytest.mark.awq_gemm
def test_awq_gemm_perf():
    def input_fn(shape, dtype, device):
        M, N, K = shape
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(K, N, dtype=dtype, device=device) * 0.1
        qweight, scales, qzeros = pack_awq_weight(w, GROUP_SIZE, dtype=dtype)
        yield x, qweight, qzeros, scales, GROUP_SIZE

    bench = AWQGemmBenchmark(
        op_name="awq_gemm",
        torch_op=torch_awq_gemm,
        input_fn=input_fn,
        gems_op=flag_gems.awq_gemm,
        dtypes=[torch.float16],
    )
    bench.run()
