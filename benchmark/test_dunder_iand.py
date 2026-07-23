import pytest

from . import base, consts, utils


@pytest.mark.dunder_iand_tensor
def test_dunder_iand_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_iand_tensor",
        torch_op=lambda a, b: a.__iand__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _input_fn_scalar(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    if cur_dtype == consts.BOOL_DTYPES[0]:
        inp2 = True
    else:
        inp2 = 0x00FF
    yield inp1, inp2


@pytest.mark.dunder_iand_scalar
def test_dunder_iand_scalar_inplace():
    bench = base.GenericBenchmark(
        op_name="dunder_iand_scalar",
        input_fn=_input_fn_scalar,
        torch_op=lambda a, b: a.__iand__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
