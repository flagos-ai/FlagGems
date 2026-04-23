from typing import Generator

import pytest
import torch

from benchmark.attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from benchmark.performance_utils import Benchmark


class BlasBenchmark(Benchmark):
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        return None


class EinsumMatmulBenchmark(BlasBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for b, m, n, k in self.shapes:
            inp1 = torch.randn([m, k], dtype=cur_dtype, device=self.device)
            inp2 = torch.randn([k, n], dtype=cur_dtype, device=self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        return A.shape[0] * B.shape[1] * A.shape[1] * 2


def einsum_matmul_op(A, B):
    return torch.einsum("ij,jk->ik", A, B)


@pytest.mark.einsum
def test_einsum_matmul_benchmark():
    bench = EinsumMatmulBenchmark(
        input_fn=None,
        op_name="einsum_matmul",
        torch_op=einsum_matmul_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class EinsumBmmBenchmark(BlasBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for b, m, n, k in self.shapes:
            inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=self.device)
            inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        return A.shape[0] * A.shape[1] * B.shape[2] * A.shape[2] * 2


def einsum_bmm_op(A, B):
    return torch.einsum("bij,bjk->bik", A, B)


@pytest.mark.einsum
def test_einsum_bmm_benchmark():
    bench = EinsumBmmBenchmark(
        input_fn=None,
        op_name="einsum_bmm",
        torch_op=einsum_bmm_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
