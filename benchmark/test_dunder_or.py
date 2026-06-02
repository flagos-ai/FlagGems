import pytest

from . import base, consts, utils


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0x00FF


@pytest.mark.dunder_or_tensor
def test_dunder_or_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_or_tensor",
        torch_op=lambda a, b: a | b,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.dunder_or_scalar
def test_dunder_or_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="dunder_or_scalar",
        torch_op=lambda a, b: a | b,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
