from typing import Generator

import pytest
import torch

from benchmark.attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from benchmark.performance_utils import Benchmark


class EinsumBenchmark(Benchmark):
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, batched=False, **kwargs):
        self.batched = batched
        super().__init__(*args, **kwargs)

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for b, m, n, k in self.shapes:
            if self.batched:
                inp1 = torch.randn([b, m, k], dtype=dtype, device=self.device)
                inp2 = torch.randn([b, k, n], dtype=dtype, device=self.device)
            else:
                inp1 = torch.randn([m, k], dtype=dtype, device=self.device)
                inp2 = torch.randn([k, n], dtype=dtype, device=self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        if self.batched:
            return A.shape[0] * A.shape[1] * B.shape[2] * A.shape[2] * 2
        return A.shape[0] * B.shape[1] * A.shape[1] * 2


def dot_input_fn(shape, dtype, device):
    (n,) = shape
    yield torch.randn(n, dtype=dtype, device=device), torch.randn(
        n, dtype=dtype, device=device
    )


def outer_input_fn(shape, dtype, device):
    m, n = shape
    yield torch.randn(m, dtype=dtype, device=device), torch.randn(
        n, dtype=dtype, device=device
    )


def unary_2d_input_fn(shape, dtype, device):
    m, n = shape
    yield (torch.randn(m, n, dtype=dtype, device=device),)


def unary_3d_input_fn(shape, dtype, device):
    m, n, k = shape
    yield (torch.randn(m, n, k, dtype=dtype, device=device),)


def ellipsis_input_fn(shape, dtype, device):
    b, h, m, k, n = shape
    yield torch.randn(b, h, m, k, dtype=dtype, device=device), torch.randn(
        b, h, k, n, dtype=dtype, device=device
    )


@pytest.mark.einsum
def test_einsum_matmul():
    bench = EinsumBenchmark(
        input_fn=None,
        op_name="einsum_matmul",
        torch_op=lambda A, B: torch.einsum("ij,jk->ik", A, B),
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.einsum
def test_einsum_bmm():
    bench = EinsumBenchmark(
        input_fn=None,
        op_name="einsum_bmm",
        torch_op=lambda A, B: torch.einsum("bij,bjk->bik", A, B),
        dtypes=FLOAT_DTYPES,
        batched=True,
    )
    bench.run()


@pytest.mark.einsum
def test_einsum_dot():
    bench = Benchmark(
        input_fn=dot_input_fn,
        op_name="einsum_dot",
        torch_op=lambda A, B: torch.einsum("i,i->", A, B),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(1024,), (4096,), (65536,)]
    bench.run()


@pytest.mark.einsum
def test_einsum_outer():
    bench = Benchmark(
        input_fn=outer_input_fn,
        op_name="einsum_outer",
        torch_op=lambda A, B: torch.einsum("i,j->ij", A, B),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_trace():
    bench = Benchmark(
        input_fn=unary_2d_input_fn,
        op_name="einsum_trace",
        torch_op=lambda A: torch.einsum("ii->", A),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_diagonal():
    bench = Benchmark(
        input_fn=unary_2d_input_fn,
        op_name="einsum_diagonal",
        torch_op=lambda A: torch.einsum("ii->i", A),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_transpose():
    bench = Benchmark(
        input_fn=unary_2d_input_fn,
        op_name="einsum_transpose",
        torch_op=lambda A: torch.einsum("ij->ji", A),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_sum_all():
    bench = Benchmark(
        input_fn=unary_3d_input_fn,
        op_name="einsum_sum_all",
        torch_op=lambda A: torch.einsum("ijk->", A),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(64, 64, 64), (128, 128, 128)]
    bench.run()


@pytest.mark.einsum
def test_einsum_sum_dim():
    bench = Benchmark(
        input_fn=unary_3d_input_fn,
        op_name="einsum_sum_dim",
        torch_op=lambda A: torch.einsum("ijk->j", A),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(64, 64, 64), (128, 128, 128)]
    bench.run()


@pytest.mark.einsum
def test_einsum_ellipsis():
    bench = Benchmark(
        input_fn=ellipsis_input_fn,
        op_name="einsum_ellipsis",
        torch_op=lambda A, B: torch.einsum("...ij,...jk->...ik", A, B),
        dtypes=FLOAT_DTYPES,
    )
    bench.shapes = [(2, 4, 64, 64, 128), (2, 8, 128, 128, 256)]
    bench.run()
