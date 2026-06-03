from typing import Generator

import pytest
import torch

from . import base, consts, utils


class ReplicationPad3dBackwardBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            # Skip non-5D shapes
            if len(shape) != 5:
                continue
            x = utils.generate_tensor_input(shape, dtype, self.device)
            padding = (1, 1, 1, 1, 1, 1)
            D, H, W = shape[2], shape[3], shape[4]
            grad_output_shape = (shape[0], shape[1], D + 2, H + 2, W + 2)
            grad_output = torch.randn(
                grad_output_shape, dtype=dtype, device=self.device
            )
            yield grad_output, x, padding


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward():
    bench = ReplicationPad3dBackwardBenchmark(
        op_name="replication_pad3d_backward",
        torch_op=torch.ops.aten.replication_pad3d_backward,
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(2, 3, 8, 16, 16), (1, 2, 4, 12, 12), (4, 64, 32, 32, 32)],
    )
    bench.run()
