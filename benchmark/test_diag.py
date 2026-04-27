import random

import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    input = utils.generate_tensor_input(shape, dtype, device)
    diagonal = random.randint(-4, 4)
    yield input, {
        "diagonal": diagonal,
    },


@pytest.mark.diag
def test_diag():
    bench = utils.GenericBenchmarkExcluse3D(
        op_name="diag",
        input_fn=_input_fn,
        torch_op=torch.diag,
        dtypes=attr_utils.FLOAT_DTYPES + attr_utils.INT_DTYPES + attr_utils.BOOL_DTYPES,
    )

    bench.run()
