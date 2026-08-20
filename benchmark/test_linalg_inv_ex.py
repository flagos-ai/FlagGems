import pytest
import torch

import flag_gems

from . import base

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if VENDOR == "nvidia":
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]


class LinalgInvExBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape"
    DEFAULT_DTYPES = _TEST_DTYPES

    def set_shapes(self, shape_file_path=None):
        # Always use custom square matrix shapes for matrix inversion
        self.shapes = [
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (4, 16, 16),
            (16, 32, 32),
        ]

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            n = inp_shape[-1]
            # Generate well-conditioned invertible matrix using diagonal dominance
            # Use CPU tensor creation then move to device to avoid FlagGems
            # operator interference during input generation
            A = torch.randn(inp_shape, dtype=dtype, device="cpu")
            for i in range(n):
                A[..., i, i] = A[..., i, i] + n
            A = A.to(self.device)
            yield A, {}


@pytest.mark.linalg_inv_ex
def test_linalg_inv_ex():
    bench = LinalgInvExBenchmark(
        op_name="linalg_inv_ex",
        torch_op=torch.linalg.inv_ex,
        dtypes=_TEST_DTYPES,
    )
    bench.set_gems(flag_gems.linalg_inv_ex)
    bench.run()
