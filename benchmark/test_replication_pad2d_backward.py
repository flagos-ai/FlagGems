"""Performance benchmark for ``aten::replication_pad2d_backward``."""

import pytest
import torch

from . import base, consts


class ReplicationPad2dBackwardBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (1, 16, 32, 32),
        (4, 32, 64, 64),
        (8, 64, 128, 128),
        (16, 64, 256, 256),
    ]
    DEFAULT_SHAPE_DESC = "N, C, H, W"

    def get_input_iter(self, dtype):
        pad_left, pad_right, pad_top, pad_bottom = 2, 3, 1, 4
        for shape in self.shapes:
            inp = torch.randn(shape, device=self.device, dtype=dtype)
            out_shape = (
                shape[:-2]
                + (shape[-2] + pad_top + pad_bottom,)
                + (shape[-1] + pad_left + pad_right,)
            )
            grad_output = torch.randn(out_shape, device=self.device, dtype=dtype)
            yield grad_output, inp, [pad_left, pad_right, pad_top, pad_bottom]


@pytest.mark.replication_pad2d_backward
def test_replication_pad2d_backward():
    bench = ReplicationPad2dBackwardBenchmark(
        op_name="replication_pad2d_backward",
        torch_op=torch.ops.aten.replication_pad2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
