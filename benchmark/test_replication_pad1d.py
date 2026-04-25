import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReplicationPad1dBenchmark(utils.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((2, 3, 7), (1, 2)),
            ((4, 16, 64), (3, 1)),
            ((8, 32, 256), (1, 2)),
            ((32, 256), (3, 1)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.replication_pad1d
def test_replication_pad1d():
    bench = ReplicationPad1dBenchmark(
        op_name="replication_pad1d",
        torch_op=torch.ops.aten.replication_pad1d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
