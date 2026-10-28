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


def _spdiags_torch_cpu(diagonals, offsets, shape, _diagonals_dev, _offsets_dev):
    """CPU baseline for ``_spdiags`` since torch has no CUDA implementation.

    Only the CPU reference inputs are consumed; the device copies are ignored
    (they are yielded so the FlagGems op can reuse the same prepared inputs).
    """
    return torch.ops.aten._spdiags(diagonals, offsets, shape)


def _spdiags_gems(_diagonals_cpu, _offsets_cpu, shape, diagonals, offsets):
    """FlagGems path, consuming the pre-prepared device inputs only."""
    return flag_gems.spdiags(diagonals, offsets, shape)


class SpdiagsBenchmark(base.Benchmark):
    """Benchmark for ``_spdiags``.

    The torch baseline runs on CPU (no CUDA kernel exists), while the FlagGems
    implementation runs its Triton kernel on the target device. Both input sets
    are prepared up-front in ``get_input_iter`` (outside the timed region) so
    neither scope pays a host-device transfer: the reference receives CPU
    tensors directly and the FlagGems op receives device tensors directly.
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
            # Prepare the CPU reference inputs here, outside the timed region.
            yield diagonals.cpu(), offsets.cpu(), [nrows, ncols], diagonals, offsets


@pytest.mark.spdiags
def test_spdiags():
    bench = SpdiagsBenchmark(
        op_name="spdiags",
        torch_op=_spdiags_torch_cpu,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=_spdiags_gems,
    )
    bench.run()
