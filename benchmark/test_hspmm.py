import pytest
import torch

from . import base


def _make_sparse_coo(shape, nnz, dtype, device):
    """Create a random coalesced sparse COO tensor."""
    M, K = shape
    if nnz == 0:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        values = torch.zeros(0, dtype=dtype, device=device)
    else:
        row_indices = torch.randint(0, M, (nnz,), device=device)
        col_indices = torch.randint(0, K, (nnz,), device=device)
        indices = torch.stack([row_indices, col_indices], dim=0)
        values = torch.randn(nnz, dtype=dtype, device=device)

    sp = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)
    return sp.coalesce()


def hspmm_input_fn(b, m, n, k, dtype, device, b_column_major):
    """
    Generate inputs for hspmm benchmark.
    For hspmm: b is unused, m=M, k=K, n=N.
    We encode nnz in b to reuse BlasBenchmark infrastructure.
    """
    del b_column_major  # Unused for sparse ops
    nnz = b  # Reuse b dimension to encode nnz
    mat1 = _make_sparse_coo((m, k), nnz, dtype, device)
    mat2 = torch.randn(k, n, dtype=dtype, device=device)
    yield mat1, mat2


class HspmmBenchmark(base.BlasBenchmark):
    """Custom benchmark for hspmm with sparse-specific shapes."""

    def set_more_shapes(self):
        """Define (nnz, M, N, K) tuples for benchmark."""
        # Format: (nnz, M, N, K) where nnz is encoded in the b dimension
        return [
            (256, 64, 32, 64),  # Small
            (5000, 512, 64, 512),  # Medium
            (20000, 1024, 128, 1024),  # Large
            (100000, 4096, 256, 1024),  # Very large
        ]

    def get_tflops(self, op, *args, **kwargs):
        """Compute TFLOPS for sparse matmul."""
        mat1, mat2 = args
        nnz = mat1._nnz()
        K, N = mat2.shape
        # Each nnz performs one scalar-vector multiply: N muls
        # Then results are reduced (summed) per output row
        # Approximate as nnz * N multiplies
        flops = nnz * N * 2  # mul + implicit add in reduction
        return flops


@pytest.mark.hspmm
def test_hspmm():
    """Benchmark hspmm operator."""
    bench = HspmmBenchmark(
        op_name="hspmm",
        input_fn=hspmm_input_fn,
        torch_op=torch.hspmm,
        # CUDA aten::hspmm only supports float32/float64 (fp16/bf16 not implemented)
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
