from typing import Generator

import pytest
import torch

from . import base, consts, utils


class ReplicationPad2dBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape, padding in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device)
            yield input, padding

    def get_bw_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape, padding in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device)
            output_shape = (
                shape[0],
                shape[1],
                shape[2] + padding[0] + padding[1],
                shape[3] + padding[2] + padding[3],
            )
            grad_output = utils.generate_tensor_input(output_shape, dtype, self.device)
            yield grad_output, input, padding


@pytest.mark.replication_pad2d
def test_replication_pad2d():
    bench = ReplicationPad2dBenchmark(
        op_name="replication_pad2d",
        torch_op=torch.ops.aten.replication_pad2d,
        dtypes=consts.FLOAT_DTYPES,
        shapes=[
            ((16, 3, 64, 64), (1, 1, 1, 1)),
            ((16, 16, 128, 128), (2, 2, 2, 2)),
            ((32, 32, 256, 256), (3, 3, 3, 3)),
        ],
    )
    bench.run()
