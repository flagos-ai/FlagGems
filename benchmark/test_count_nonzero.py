import random

import pytest
import torch

from . import base, consts


@pytest.mark.count_nonzero
def test_count_nonzero():
    def count_nonzero_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = random.choice([None, 0, 1])

        yield inp, dim

    bench = base.GenericBenchmark2DOnly(
        input_fn=count_nonzero_input_fn,
        op_name="count_nonzero",
        torch_op=torch.count_nonzero,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
