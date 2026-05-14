import pytest
import torch

from . import base, consts


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES (Issue #2693).
@pytest.mark.div_tensor
def test_div():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_tensor_
def test_div_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor_",
        torch_op=lambda a, b: a.div_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_mode_inplace(rounding_mode):
    if rounding_mode in ("trunc", "floor"):
        pytest.xfail(
            "Operator bug: trunc/floor div kernels fail Triton compilation for float dtypes"
        )
    bench = base.BinaryPointwiseBenchmark(
        op_name=f"div_tensor_mode_({rounding_mode})",
        torch_op=lambda a, b: a.div_(b, rounding_mode=rounding_mode),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
