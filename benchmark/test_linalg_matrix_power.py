"""Performance benchmark for linalg.matrix_power.

Covers the key performance dimensions:
  - Matrix size: small (2→64) through large (128→512) — O(n³) per matmul
  - Exponent value: controls the number of matmul calls (O(log n))
  - Batch dimension: checks that batched matmul is used efficiently
"""

import pytest
import torch

from . import base

# ---------------------------------------------------------------------------
# Benchmark shapes — organised by characteristic
# ---------------------------------------------------------------------------

# Varying matrix size at fixed exponent n=5 (3 matmuls: 2 squarings + 1 accumulate)
SIZE_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]

# Varying exponent at fixed matrix size 32×32
POWER_EXPONENTS = [0, 1, 2, 3, 4, 8, 16, 32, 64]


class MatrixPowerBenchmarkBase(base.Benchmark):
    """Shared logic for all matrix_power benchmark variants."""

    torch_op = torch.ops.aten.linalg_matrix_power

    def get_input_iter(self, cur_dtype):
        raise NotImplementedError


# ===========================================================================
# Benchmark 1 — Matrix size sweep  (n=5, float32)
# ===========================================================================


class MatrixPowerSizeSweep(MatrixPowerBenchmarkBase):
    N = 5  # 3 matmuls: two squarings + one accumulate (n=5 = 101b)

    def set_shapes(self, shape_file_path=None):
        self.shapes = SIZE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A, self.N)


@pytest.mark.linalg_matrix_power
def test_benchmark_size_sweep():
    bench = MatrixPowerSizeSweep(
        op_name="linalg_matrix_power",
        torch_op=torch.ops.aten.linalg_matrix_power,
        dtypes=[torch.float32],
    )
    bench.run()


# ===========================================================================
# Benchmark 2 — Exponent sweep  (32×32, float32)
# ===========================================================================

# Map exponent → approx matmul count
EXPONENT_SHAPES = [(32, 32)]


class MatrixPowerExponentSweep(MatrixPowerBenchmarkBase):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EXPONENT_SHAPES

    def get_input_iter(self, cur_dtype):
        A = torch.randn(self.shapes[0], dtype=cur_dtype, device=self.device)
        for n in POWER_EXPONENTS:
            yield (A.clone(), n)


@pytest.mark.linalg_matrix_power
def test_benchmark_exponent_sweep():
    bench = MatrixPowerExponentSweep(
        op_name="linalg_matrix_power",
        torch_op=torch.ops.aten.linalg_matrix_power,
        dtypes=[torch.float32],
    )
    bench.run()


# ===========================================================================
# Benchmark 3 — Batched matrices  (batch=8, 16×16, n=5, float32)
# ===========================================================================

BATCH_SHAPES = [
    (8, 16, 16),
    (16, 16, 16),
    (32, 16, 16),
    (4, 8, 8),
    (8, 32, 32),
]


class MatrixPowerBatchBench(MatrixPowerBenchmarkBase):
    N = 5

    def set_shapes(self, shape_file_path=None):
        self.shapes = BATCH_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A, self.N)


@pytest.mark.linalg_matrix_power
def test_benchmark_batch():
    bench = MatrixPowerBatchBench(
        op_name="linalg_matrix_power",
        torch_op=torch.ops.aten.linalg_matrix_power,
        dtypes=[torch.float32],
    )
    bench.run()


# ===========================================================================
# Benchmark 4 — dtype sweep  (32×32, n=5)
# ===========================================================================


class MatrixPowerDtypeSweep(MatrixPowerBenchmarkBase):
    shape = (32, 32)
    N = 5

    def set_shapes(self, shape_file_path=None):
        self.shapes = [self.shape]

    def get_input_iter(self, cur_dtype):
        A = torch.randn(self.shape, dtype=cur_dtype, device=self.device)
        yield (A, self.N)


@pytest.mark.linalg_matrix_power
def test_benchmark_dtypes():
    bench = MatrixPowerDtypeSweep(
        op_name="linalg_matrix_power",
        torch_op=torch.ops.aten.linalg_matrix_power,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
