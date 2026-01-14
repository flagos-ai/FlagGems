import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark


def torch_channel_shuffle(x, groups):
    return torch.nn.ChannelShuffle(groups)(x)


def channel_shuffle_input_fn(shape, dtype, device):
    n, c, h, w, groups = shape
    x = torch.randn((n, c, h, w), dtype=dtype, device=device)
    yield x, {"groups": groups}


class ChannelShuffleBenchmark(GenericBenchmark):
    def get_input_iter(self, cur_dtype):
        shapes = [
            (2, 8, 3, 3, 2),
            (1, 12, 5, 6, 3),
            (4, 16, 32, 32, 4),
            (32, 1024, 32, 32, 8),
        ]
        for shape in shapes:
            yield from channel_shuffle_input_fn(shape, cur_dtype, self.device)


@pytest.mark.channel_shuffle
def test_channel_shuffle_perf():
    bench = ChannelShuffleBenchmark(
        input_fn=channel_shuffle_input_fn,
        op_name="channel_shuffle",
        torch_op=torch_channel_shuffle,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.channel_shuffle)
    bench.run()
