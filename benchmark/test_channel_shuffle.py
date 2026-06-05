import pytest
import torch

from . import base, consts


@pytest.mark.channel_shuffle
def test_channel_shuffle():
    def channel_shuffle_input_fn(config, dtype, device):
        shape, groups = config
        x = torch.randn(shape, dtype=dtype, device=device)
        yield x, groups

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
            for config in self.shapes:
                yield from channel_shuffle_input_fn(config, cur_dtype, self.device)

    bench = ChannelShuffleBenchmark(
        op_name="channel_shuffle",
        torch_op=torch.ops.aten.channel_shuffle,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
