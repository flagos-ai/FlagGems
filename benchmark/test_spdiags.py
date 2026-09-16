import pytest
import torch

import flag_gems

from . import base, consts

# Square matrix sizes used to build the diagonal input. For each size we build a
# tri-diagonal input (offsets -1, 0, 1) which is a common sparse pattern.
SPDIAGS_SHAPES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
]


def _spdiags_torch_cpu(diagonals, offsets, shape):
    """CPU baseline wrapper for _spdiags since torch has no CUDA implementation."""
    return torch.ops.aten._spdiags(diagonals.cpu(), offsets.cpu(), shape)


class SpdiagsBenchmark(base.Benchmark):
    """Benchmark for ``_spdiags``.

    The torch baseline runs on CPU (no CUDA kernel exists), while the FlagGems
    implementation runs its Triton kernel on the target device.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = SPDIAGS_SHAPES

    def get_input_iter(self, cur_dtype):
        for nrows, ncols in self.shapes:
            diag_len = min(nrows, ncols)
            num_diags = 3
            diagonals = torch.randn(
                (num_diags, diag_len), dtype=cur_dtype, device=self.device
            )
            offsets = torch.tensor([-1, 0, 1], dtype=torch.int64, device=self.device)
            yield diagonals, offsets, [nrows, ncols]


@pytest.mark.spdiags
def test_spdiags():
    bench = SpdiagsBenchmark(
        op_name="spdiags",
        torch_op=_spdiags_torch_cpu,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=flag_gems._spdiags,
    )
    bench.run()
