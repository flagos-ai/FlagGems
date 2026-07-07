import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn


@pytest.mark.asinh
def test_asinh():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="asinh",
        torch_op=torch.asinh,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.asinh_
def test_asinh_inplace():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="asinh_",
        torch_op=torch.asinh_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
