"""Performance benchmark for ``aten::replication_pad1d_backward``."""

import pytest
import torch

from . import base, consts


class ReplicationPad1dBackwardBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (1, 16, 128),
        (4, 32, 256),
        (8, 64, 512),
        (16, 64, 1024),
    ]
    DEFAULT_SHAPE_DESC = "N, C, L"

    def get_input_iter(self, dtype):
        pad_left, pad_right = 3, 5
        for shape in self.shapes:
            inp = torch.randn(shape, device=self.device, dtype=dtype)
            out_shape = shape[:-1] + (shape[-1] + pad_left + pad_right,)
            grad_output = torch.randn(out_shape, device=self.device, dtype=dtype)
            yield grad_output, inp, [pad_left, pad_right]


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward():
    bench = ReplicationPad1dBackwardBenchmark(
        op_name="replication_pad1d_backward",
        torch_op=torch.ops.aten.replication_pad1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
