import pytest
import torch

from . import base, consts


class ChannelShuffleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((2, 64, 32, 32), 2),
            ((2, 128, 64, 64), 4),
            ((4, 256, 128, 128), 8),
            ((8, 512, 256, 256), 16),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for (shape, groups) in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            yield x, groups


@pytest.mark.channel_shuffle
def test_channel_shuffle():
    bench = ChannelShuffleBenchmark(
        op_name="channel_shuffle",
        torch_op=lambda x, g: torch.ops.aten.channel_shuffle(x, g),
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
