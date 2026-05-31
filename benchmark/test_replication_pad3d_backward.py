"""Performance benchmark for ``aten::replication_pad3d_backward``."""

import pytest
import torch

from . import base, consts


class ReplicationPad3dBackwardBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (1, 8, 4, 16, 16),
        (2, 16, 8, 32, 32),
        (4, 32, 8, 64, 64),
        (4, 32, 16, 32, 32),
    ]
    DEFAULT_SHAPE_DESC = "N, C, D, H, W"

    def get_input_iter(self, dtype):
        pl, pr, pt, pb, pf, pbk = 2, 3, 1, 4, 1, 2
        for shape in self.shapes:
            inp = torch.randn(shape, device=self.device, dtype=dtype)
            out_shape = (
                shape[:-3]
                + (shape[-3] + pf + pbk,)
                + (shape[-2] + pt + pb,)
                + (shape[-1] + pl + pr,)
            )
            grad_output = torch.randn(out_shape, device=self.device, dtype=dtype)
            yield grad_output, inp, [pl, pr, pt, pb, pf, pbk]


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward():
    bench = ReplicationPad3dBackwardBenchmark(
        op_name="replication_pad3d_backward",
        torch_op=torch.ops.aten.replication_pad3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
