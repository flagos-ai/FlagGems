import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    elements = utils.generate_tensor_input(shape, dtype, device)
    test_elements = utils.generate_tensor_input(shape, dtype, device)

    yield elements, test_elements

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # assume_unique set to True
        uniq_elements = torch.unique(utils.generate_tensor_input(shape, dtype, device))
        uniq_test_elements = torch.unique(
            utils.generate_tensor_input(shape, dtype, device)
        )
        yield uniq_elements, uniq_test_elements, {"assume_unique": True}


@pytest.mark.isin
def test_isin():
    bench = base.GenericBenchmark2DOnly(
        op_name="isin",
        input_fn=_input_fn,
        torch_op=torch.isin,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()


def _scalar_tensor_input_fn(shape, dtype, device):
    test_elements = utils.generate_tensor_input(shape, dtype, device)
    scalar_val = int(test_elements.ravel()[0].item())

    yield scalar_val, test_elements

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        uniq_test_elements = torch.unique(
            utils.generate_tensor_input(shape, dtype, device)
        )
        scalar_val2 = int(uniq_test_elements.ravel()[0].item())
        yield scalar_val2, uniq_test_elements, {"assume_unique": True}


@pytest.mark.isin_scalar_tensor
def test_isin_scalar_tensor():
    bench = base.GenericBenchmark2DOnly(
        op_name="isin_scalar_tensor",
        input_fn=_scalar_tensor_input_fn,
        torch_op=torch.isin,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()


def _input_fn_tensor_scalar(shape, dtype, device):
    elements = utils.generate_tensor_input(shape, dtype, device)
    test_value = 42

    yield elements, test_value

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        uniq_elements = torch.unique(utils.generate_tensor_input(shape, dtype, device))
        yield uniq_elements, test_value, {"assume_unique": True}


@pytest.mark.isin_tensor_scalar
def test_isin_tensor_scalar():
    bench = base.GenericBenchmark2DOnly(
        op_name="isin_tensor_scalar",
        input_fn=_input_fn_tensor_scalar,
        torch_op=torch.isin,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()
