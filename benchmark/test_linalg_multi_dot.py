import pytest
import torch

from . import base

# Using subset of consts.FLOAT_DTYPES: float16 excluded because flag_gems mm
# with float16 triggers cluster_remote_mm using tle_exp.gpu.alloc which requires
# make_nv_mma_shared_encoding_attr not available in this Triton version.
LINALG_MULTI_DOT_DTYPES = [torch.float32, torch.bfloat16]

# Custom shapes for linalg_multi_dot benchmark: 3 matrices chained as
# (m,n) @ (n,k) @ (k,n). 4-tuple (B, M, N, K) maps to B=unused, M=m, N=n, K=k.
# Not sourced from core_shapes.yaml because linalg_multi_dot is a meta-operator
# requiring shape triples, not standard BLAS (B,M,N,K) pairs.
_SHAPES = [
    (2, 128, 256, 256),
    (2, 256, 512, 256),
    (2, 512, 1024, 512),
    (2, 384, 384, 384),
    (2, 1024, 1024, 1024),
    (2, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
]


class LinalgMultiDotBenchmark(base.BlasBenchmark):
    def get_input_iter(self, dtype):
        for b, m, n, k in _SHAPES:
            inp1 = torch.randn(m, n, dtype=dtype, device=self.device)
            inp2 = torch.randn(n, k, dtype=dtype, device=self.device)
            inp3 = torch.randn(k, n, dtype=dtype, device=self.device)
            yield [inp1, inp2, inp3],

    def get_tflops(self, op, *args, **kwargs):
        # args[0] is [inp1(m,n), inp2(n,k), inp3(k,n)]
        # FLOPs: (m,n)@(n,k) + (m,k)@(k,n) = 2*m*k*n + 2*m*n*k = 4*m*n*k
        tensors = args[0]
        m, n = tensors[0].shape
        k = tensors[1].shape[1]
        return 4 * m * n * k


@pytest.mark.linalg_multi_dot
def test_linalg_multi_dot():
    bench = LinalgMultiDotBenchmark(
        op_name="linalg_multi_dot",
        input_fn=None,
        torch_op=torch.linalg.multi_dot,
        dtypes=LINALG_MULTI_DOT_DTYPES,
    )
    bench.run()
