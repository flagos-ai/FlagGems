import pytest
import torch

import flag_gems

from . import base

# Shapes for linalg_eig benchmark - only 2x2 is implemented
# Note: The Triton kernel only handles 2x2 matrices; larger matrices
# are delegated to torch.ops.aten.linalg_eig (cuSOLVER)
LINALG_EIG_SHAPES = [
    (2, 2),
]


class LinalgEigBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_EIG_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


# linalg_eig only supports float32 — eigenvalue decomposition requires
# float32 precision; Half/BFloat16 are not supported by torch.linalg.eig
@pytest.mark.linalg_eig
def test_linalg_eig():
    bench = LinalgEigBenchmark(
        op_name="linalg_eig",
        torch_op=torch.linalg.eig,
        # linalg_eig only supports float32 — Half/BFloat16 not supported
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_eig)
    bench.run()
