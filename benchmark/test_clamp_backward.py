from typing import Generator

import pytest
import torch

from . import base, consts, utils


class ClampTensorBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            min_val = torch.randn_like(inp)
            max_val = min_val + torch.abs(torch.randn_like(inp)) + 0.5
            grad_out = torch.randn_like(inp)
            yield grad_out, inp, min_val, max_val


@pytest.mark.clamp_backward
def test_clamp_tensor_backward():
    bench = ClampTensorBackwardBenchmark(
        op_name="clamp.Tensor_backward",
        torch_op=torch.ops.aten.clamp.Tensor_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class ClampMinTensorBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            min_val = torch.randn_like(inp)
            grad_out = torch.randn_like(inp)
            yield grad_out, inp, min_val


@pytest.mark.clamp_backward
def test_clamp_min_tensor_backward():
    bench = ClampMinTensorBackwardBenchmark(
        op_name="clamp_min.Tensor_backward",
        torch_op=torch.ops.aten.clamp_min.Tensor_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class ClampMaxTensorBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            max_val = torch.randn_like(inp)
            grad_out = torch.randn_like(inp)
            yield grad_out, inp, max_val


@pytest.mark.clamp_backward
def test_clamp_max_tensor_backward():
    bench = ClampMaxTensorBackwardBenchmark(
        op_name="clamp_max.Tensor_backward",
        torch_op=torch.ops.aten.clamp_max.Tensor_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
