from typing import Generator

import pytest
import torch

from . import base, consts, utils


@pytest.mark.swish
def test_swish():
    bench = base.UnaryPointwiseBenchmark(
        op_name="swish", torch_op=torch.nn.functional.silu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.swish_
def test_swish_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="swish_",
        torch_op=lambda a: torch.nn.functional.silu(a, inplace=True),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


class SwishBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            yield grad_out, inp


@pytest.mark.swish_backward
def test_swish_backward():
    bench = SwishBackwardBenchmark(
        op_name="swish_backward",
        torch_op=torch.ops.aten.silu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
