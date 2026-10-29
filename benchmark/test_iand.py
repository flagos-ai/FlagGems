import pytest

from . import base, consts, utils


@pytest.mark.bitwise_and_tensor_
def test_iand_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_tensor_",
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


def _case_fn_scalar(shape, cur_dtype):
    scalar = True if cur_dtype == consts.BOOL_DTYPES[0] else 0x00FF
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"scalar": scalar},
        builder_args=(shape,),
    )


def _build_inputs_fn_scalar(plan, cur_dtype, device):
    return next(_input_fn_scalar(plan.builder_args[0], cur_dtype, device))


@pytest.mark.bitwise_and_scalar_
def test_iand_scalar_inplace():
    bench = base.GenericBenchmark(
        op_name="bitwise_and_scalar_",
        case_fn=_case_fn_scalar,
        build_inputs_fn=_build_inputs_fn_scalar,
        torch_op=lambda a, b: a.__iand__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
