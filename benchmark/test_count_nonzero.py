import random

import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.count_nonzero
def test_count_nonzero():
    def count_nonzero_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = random.choice([None, 0, 1])

        yield inp, dim

    bench = utils.GenericBenchmark2DOnly(
        input_fn=count_nonzero_input_fn,
        op_name="count_nonzero",
        torch_op=torch.count_nonzero,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
