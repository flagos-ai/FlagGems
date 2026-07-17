import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.pow_tensor_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_tensor_tensor():
    bench = base.ScalarBinaryPointwiseBenchmark(
        op_name="pow_tensor_tensor",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pow_tensor_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="pow_tensor_tensor_",
        torch_op=lambda a, b: a.pow_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.pow_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_scalar():
    bench = base.ScalarBinaryPointwiseBenchmark(
        op_name="pow_scalar",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def pow_tensor_scalar_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    scalar = 0.001
    yield inp, scalar


@pytest.mark.pow_tensor_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_tensor_scalar():
    bench = base.GenericBenchmark(
        input_fn=pow_tensor_scalar_input_fn,
        op_name="pow_tensor_scalar",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pow_tensor_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_tensor_scalar_inplace():
    bench = base.GenericBenchmark(
        input_fn=pow_tensor_scalar_input_fn,
        op_name="pow_tensor_scalar_",
        torch_op=lambda a, b: a.pow_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
