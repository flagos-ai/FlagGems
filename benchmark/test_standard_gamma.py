import pytest
import torch

import flag_gems

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


@pytest.mark.standard_gamma_backward
def test_standard_gamma_backward():
    """Time the alpha backward of ``torch._standard_gamma`` through autograd.

    ``is_backward=True`` makes the timed callable
    ``torch.autograd.grad(sample, alpha, grad_output, retain_graph=True)``: the
    forward runs once outside the timer and only the backward is measured.  The
    unary shapes come from ``UnaryPointwiseBenchmark`` (the class
    ``test_standard_gamma`` already uses) filtered by the shared
    ``MAX_BACKWARD_ELEMENTS`` budget, so the case set is the repository default.
    """
    bench = base.UnaryPointwiseBenchmark(
        op_name="standard_gamma",
        torch_op=torch._standard_gamma,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.standard_gamma)
    bench.run()
