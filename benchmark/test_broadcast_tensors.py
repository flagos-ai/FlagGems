from typing import Generator

import pytest
import torch

from . import base, consts, utils


class BroadcastTensorsBenchmark(base.Benchmark):
    # broadcast_tensors accepts multiple tensors of different shapes,
    # so shapes here represent the target broadcast output shape.
    DEFAULT_SHAPE_DESC = "target shape"

    def set_shapes(self, shape_file_path=None):
        # Representative sizes covering 2D, 3D and a range of element counts
        # for broadcast_tensors performance measurement.
        self.shapes = [
            (64, 64),
            (256, 256),
            (4096, 4096),
            (64, 512, 512),
            (1024, 1024, 1024),
        ]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            # Produce two tensors with different shapes that require actual broadcasting
            if len(shape) >= 2:
                shape_a = list(shape)
                shape_a[0] = 1
                shape_b = list(shape)
                shape_b[1] = 1
                inp1 = utils.generate_tensor_input(tuple(shape_a), dtype, self.device)
                inp2 = utils.generate_tensor_input(tuple(shape_b), dtype, self.device)
            else:
                inp1 = utils.generate_tensor_input(shape, dtype, self.device)
                inp2 = utils.generate_tensor_input((1,), dtype, self.device)
            yield inp1, inp2


@pytest.mark.broadcast_tensors
def test_broadcast_tensors():
    bench = BroadcastTensorsBenchmark(
        op_name="broadcast_tensors",
        torch_op=torch.broadcast_tensors,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
