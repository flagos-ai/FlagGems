import pytest
import torch

from . import base, consts, utils

# Custom shapes that include large tensors where Triton nonzero excels
ARGWHERE_SHAPES = [
    (4096, 4096),
    (1024, 65536),
    (8192, 8192),
    (2048, 65536),
    (4096, 65536),
]


@pytest.mark.argwhere
def test_argwhere():
    bench = base.GenericBenchmark2DOnly(
        input_fn=utils.unary_input_fn,
        op_name="argwhere",
        torch_op=torch.argwhere,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = ARGWHERE_SHAPES
    bench.run()
