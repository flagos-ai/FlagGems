from typing import Generator

import pytest
import torch

from . import base, consts, utils


@pytest.mark.silu
def test_silu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="silu", torch_op=torch.nn.functional.silu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.silu_
def test_silu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="silu_",
        torch_op=lambda a: torch.nn.functional.silu(a, inplace=True),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


class SiluBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            yield grad_out, inp


@pytest.mark.silu_backward
def test_silu_backward():
    bench = SiluBackwardBenchmark(
        op_name="silu_backward",
        torch_op=torch.ops.aten.silu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
