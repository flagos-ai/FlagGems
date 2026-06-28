import pytest
import torch

from . import base

# triton's div_rz and trunc only support float32 and float64,
# float16/bf16 will cause CompilationError.
TRUNC_DIVIDE_DTYPES = [torch.float32]


@pytest.mark.trunc_divide
def test_trunc_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="trunc_divide",
        torch_op=lambda a, b: torch.div(a, b, rounding_mode="trunc"),
        dtypes=TRUNC_DIVIDE_DTYPES,
    )
    bench.run()


@pytest.mark.trunc_divide_
def test_trunc_divide_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="trunc_divide_",
        torch_op=lambda a, b: a.div_(b, rounding_mode="trunc"),
        dtypes=TRUNC_DIVIDE_DTYPES,
        is_inplace=True,
    )
    bench.run()
