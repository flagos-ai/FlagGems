import pytest
import torch

from . import base, consts


@pytest.mark.trunc_divide
def test_trunc_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="trunc_divide",
        torch_op=lambda a, b: torch.div(a, b, rounding_mode="trunc"),
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()


@pytest.mark.trunc_divide_
def test_trunc_divide_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="trunc_divide_",
        torch_op=lambda a, b: a.div_(b, rounding_mode="trunc"),
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
        is_inplace=True,
    )
    bench.run()
