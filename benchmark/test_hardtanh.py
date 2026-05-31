from typing import Generator

import pytest
import torch

from . import base, consts, utils


class HardTanhBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device)
            yield input, -1.0, 1.0


class HardTanhBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device)
            grad_output = utils.generate_tensor_input(shape, dtype, self.device)
            yield grad_output, input, -1.0, 1.0


@pytest.mark.hardtanh
def test_hardtanh():
    bench = HardTanhBenchmark(
        op_name="hardtanh",
        torch_op=torch.ops.aten.hardtanh,
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(1024,), (4096,), (16384,), (65536,)],
    )
    bench.run()


@pytest.mark.hardtanh_backward
def test_hardtanh_backward():
    bench = HardTanhBackwardBenchmark(
        op_name="hardtanh_backward",
        torch_op=torch.ops.aten.hardtanh_backward,
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(1024,), (4096,), (16384,), (65536,)],
    )
    bench.run()
