from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


class GemmBenchmark(base.GenericBenchmark2DOnly):
    """
    benchmark for gemm, test shape compatable with to mm
    """

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, k in self.shapes:
            yield from self.input_fn(1, m, m, k, cur_dtype, self.device, False)

        # if Config.bench_level == BenchLevel.COMPREHENSIVE:
        #     for m, k in self.shapes:
        #         yield from self.input_fn(1, m, m, k, cur_dtype, self.device, True)

    def set_more_shapes(self):
        large_k_shapes = [
            (8, 1848, 1536, 151936),
            (8, 1848, 1536, 128256),
            (8, 1848, 1536, 152064),
            (8, 4096, 1, 152064),
        ]

        return large_k_shapes

    def get_tflops(self, op, *args, **kwargs):
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2
        # shape(m,n)(n,p)
        # total_flops mxpx(2n+1)
        return total_flops


@pytest.mark.gemm
def test_gemm_benchmark():
    def gemm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
        inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
        if b_column_major:
            inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
            yield inp1, inp2.t()
        else:
            inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
            yield inp1, inp2

    bench = GemmBenchmark(
        input_fn=gemm_input_fn,
        op_name="gemm",
        torch_op=torch.Tensor.mm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.ops.gemm)
    bench.run()
