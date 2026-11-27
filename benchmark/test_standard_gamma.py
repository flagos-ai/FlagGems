import pytest
import torch

from . import base, consts


@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "aten::_standard_gamma has no GCU kernel; the CPU fallback raises "
        "\"gamma_cpu\" not implemented for 'Half', so the native baseline "
        "cannot be timed"
    ),
)
@pytest.mark.standard_gamma
def test_standard_gamma():
    bench = base.UnaryPointwiseBenchmark(
        op_name="standard_gamma",
        torch_op=torch._standard_gamma,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
