import pytest
import torch

from . import base, consts

# Standard shapes: (num_blocks, block_size)
BLOCK_DIAG_SHAPES = [
    (4, 64),
    (8, 128),
    (16, 64),
    (4, 256),
    (8, 256),
]


def _torch_block_diag(tensors):
    """Wrapper to call torch.block_diag with a list of tensors."""
    return torch.block_diag(*tensors)


class BlockDiagBenchmark(base.Benchmark):
    """Benchmark for block_diag."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = BLOCK_DIAG_SHAPES[:]
        self.shape_desc = "num_blocks, block_size"

    def get_input_iter(self, cur_dtype):
        for num_blocks, block_size in self.shapes:
            blocks = [
                torch.randn(
                    (block_size, block_size), dtype=cur_dtype, device=self.device
                )
                for _ in range(num_blocks)
            ]
            yield (blocks,)


@pytest.mark.block_diag
def test_block_diag():
    bench = BlockDiagBenchmark(
        op_name="block_diag",
        torch_op=_torch_block_diag,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
