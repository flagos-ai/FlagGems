from typing import Generator

import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


@pytest.mark.elu
def test_elu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="elu", torch_op=torch.nn.functional.elu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.elu_
def test_elu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="elu_",
        torch_op=torch.nn.functional.elu_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


class EluBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            alpha = 1.0
            scale = 1.0
            input_scale = 1.0
            is_result = False

            yield grad_out, alpha, scale, input_scale, is_result, inp


@pytest.mark.elu_backward
def test_elu_backward():
    bench = EluBackwardBenchmark(
        op_name="elu_backward",
        torch_op=torch.ops.aten.elu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
