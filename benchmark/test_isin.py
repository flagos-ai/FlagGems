import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    elements = utils.generate_tensor_input(shape, dtype, device)
    test_elements = utils.generate_tensor_input(shape, dtype, device)

    yield elements, test_elements

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        # assume_unique set to True
        uniq_elements = torch.unique(utils.generate_tensor_input(shape, dtype, device))
        uniq_test_elements = torch.unique(
            utils.generate_tensor_input(shape, dtype, device)
        )
        yield uniq_elements, uniq_test_elements, {"assume_unique": True}


@pytest.mark.isin
def test_isin():
    bench = utils.GenericBenchmark2DOnly(
        op_name="isin",
        input_fn=_input_fn,
        torch_op=torch.isin,
        dtypes=attr_utils.INT_DTYPES,
    )

    bench.run()
