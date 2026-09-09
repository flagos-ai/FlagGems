import pytest
import torch

from . import base, consts

REPLICATION_PAD2D_BACKWARD_SHAPES = [
    (1, 3, 256, 256),
    (1, 3, 640, 640),
    (1, 3, 1024, 1024),
    (1, 64, 128, 128),
    (1, 64, 256, 256),
    (1, 64, 512, 512),
    (1, 128, 64, 64),
    (1, 256, 32, 32),
    (4, 8, 256, 256),
    (8, 64, 128, 128),
    (8, 128, 1024, 1024),
    (3, 128, 128),
    (1, 32, 1, 128),
]

REPLICATION_PAD2D_BACKWARD_PADDINGS = [
    (1, 1, 1, 1),
    (1, 2, 3, 4),
    (3, 0, 0, 3),
    (2, 2, 2, 2),
]


class ReplicationPad2dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (shape, padding)
            for shape in REPLICATION_PAD2D_BACKWARD_SHAPES
            for padding in REPLICATION_PAD2D_BACKWARD_PADDINGS
        ]

    def get_input_iter(self, cur_dtype):
        for shape, padding in self.shapes:
            pad_left, pad_right, pad_top, pad_bottom = padding

            x = torch.randn(shape, dtype=cur_dtype, device=self.device)

            if x.ndim == 4:
                N, C, H, W = x.shape
                grad_shape = (N, C, H + pad_top + pad_bottom, W + pad_left + pad_right)
            else:
                C, H, W = x.shape
                grad_shape = (C, H + pad_top + pad_bottom, W + pad_left + pad_right)

            grad_output = torch.ones(grad_shape, dtype=cur_dtype, device=self.device)
            yield grad_output, x, padding


@pytest.mark.replication_pad2d_backward
def test_replication_pad2d_backward():
    bench = ReplicationPad2dBackwardBenchmark(
        op_name="replication_pad2d_backward",
        torch_op=torch.ops.aten.replication_pad2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
