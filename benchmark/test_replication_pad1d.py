import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReplicationPad1dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((2, 3, 7), (1, 2)),
            ((4, 16, 64), (3, 1)),
            ((8, 32, 256), (1, 2)),
            ((32, 256), (3, 1)),
        ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.replication_pad1d
def test_replication_pad1d():
    bench = ReplicationPad1dBenchmark(
        op_name="replication_pad1d",
        torch_op=torch.ops.aten.replication_pad1d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
