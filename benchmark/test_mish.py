from typing import Generator

import pytest
import torch

from . import base, consts, utils


class MishBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device)
            yield input

    def get_bw_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device, requires_grad=True)
            grad_output = utils.generate_tensor_input(shape, dtype, self.device)
            yield grad_output, input


@pytest.mark.mish
def test_mish():
    bench = MishBenchmark(
        op_name="mish",
        torch_op=torch.ops.aten.mish,
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(256,), (1024,), (4096,), (16384,)],
    )
    bench.run()


@pytest.mark.mish_backward
def test_mish_backward():
    bench = MishBenchmark(
        op_name="mish_backward",
        torch_op=lambda x: torch.ops.aten.mish(x).backward(torch.ones_like(x)),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(256,), (1024,), (4096,), (16384,)],
    )
    bench.run()
