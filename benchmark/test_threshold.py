import pytest
import torch

import flag_gems

from . import base, consts, utils

vendor_name = flag_gems.vendor_name


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 3.14, 2.71


@pytest.mark.threshold
def test_threshold():
    bench = base.GenericBenchmark(
        op_name="threshold",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.threshold,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _threshold_backward_input_fn(shape, cur_dtype, device):
    grad_output = utils.generate_tensor_input(shape, cur_dtype, device)
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield grad_output, inp, 3.14


@pytest.mark.threshold_backward
def test_threshold_backward():
    bench = base.GenericBenchmark(
        op_name="threshold_backward",
        input_fn=_threshold_backward_input_fn,
        torch_op=torch.ops.aten.threshold_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
