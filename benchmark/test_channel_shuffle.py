import pytest
import torch

from . import base, consts


class ChannelShuffleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative shapes covering small to medium NCHW inputs for channel shuffle
        self.shapes = [
            ((1, 4, 2, 2), 2),
            ((2, 8, 4, 4), 4),
            ((4, 16, 8, 8), 4),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape, groups in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x, groups


@pytest.mark.channel_shuffle
def test_channel_shuffle():
    bench = ChannelShuffleBenchmark(
        op_name="channel_shuffle",
        torch_op=torch.ops.aten.channel_shuffle,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
