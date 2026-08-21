from typing import Generator

import pytest
import torch

from . import base, consts, utils


class WhereSelfBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            condition = utils.generate_tensor_input(shape, dtype, self.device) > 0
            self = utils.generate_tensor_input(shape, dtype, self.device)
            other = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(self)
            yield grad_out, condition, self, other


@pytest.mark.where_backward
def test_where_self_backward():
    bench = WhereSelfBackwardBenchmark(
        op_name="where.Self_backward",
        torch_op=torch.ops.aten.where.Self_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class WhereOtherBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            condition = utils.generate_tensor_input(shape, dtype, self.device) > 0
            self = utils.generate_tensor_input(shape, dtype, self.device)
            other = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(self)
            yield grad_out, condition, self, other


@pytest.mark.where_backward
def test_where_other_backward():
    bench = WhereOtherBackwardBenchmark(
        op_name="where.Other_backward",
        torch_op=torch.ops.aten.where.Other_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
