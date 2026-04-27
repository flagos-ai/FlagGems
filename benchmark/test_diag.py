import random

import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def _input_fn(shape, dtype, device):
    input = utils.generate_tensor_input(shape, dtype, device)
    diagonal = random.randint(-4, 4)
    yield input, {
        "diagonal": diagonal,
    },


@pytest.mark.diag
def test_diag():
    bench = base.GenericBenchmarkExcluse3D(
        op_name="diag",
        input_fn=_input_fn,
        torch_op=torch.diag,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )

    bench.run()
