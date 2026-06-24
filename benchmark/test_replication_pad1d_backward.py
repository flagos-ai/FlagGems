import pytest
import torch

from . import base, consts

# (batch, width) pairs covering small to medium tensors for pad1d backward
REPLICATION_PAD1D_BACKWARD_SHAPES = [
    (2, 3),
    (4, 8),
    (8, 16),
    (1, 32),
    (4, 64),
    (8, 128),
]


class ReplicationPad1dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = REPLICATION_PAD1D_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if len(shape) == 2:
                B, W = shape
            else:
                B, W = 1, shape[0]
            padding = (1, 2)
            x = torch.randn(B, W, dtype=cur_dtype, device=self.device)
            # Compute forward to get output size
            padded = torch.ops.aten.replication_pad1d(x, padding)
            W_out = padded.shape[-1]
            grad = torch.ones(B, W_out, dtype=cur_dtype, device=self.device)
            yield grad, x, padding


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward():
    bench = ReplicationPad1dBackwardBenchmark(
        op_name="replication_pad1d_backward",
        torch_op=torch.ops.aten.replication_pad1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
