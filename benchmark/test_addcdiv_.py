import pytest
import torch

import flag_gems

from . import base, consts, utils


def addcdiv__input_fn(shape, dtype, device):
    # For in-place addcdiv_, we need to yield the arguments for tensor.addcdiv_() method call
    # The input function format: (inp1, inp2, inp3) + kwargs
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)

    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.addcdiv_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_addcdiv_():
    bench = base.GenericBenchmark(
        op_name="addcdiv_",
        torch_op=torch.Tensor.addcdiv_,
        input_fn=addcdiv__input_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
