import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


class EmbeddingBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield {"input": inp},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {"input": inp, "offset": 1, "dim1": 0, "dim2": -1},


@pytest.mark.diag_embed
def test_diag_embed():
    bench = EmbeddingBenchmark(
        op_name="diag_embed",
        input_fn=_input_fn,
        torch_op=torch.diag_embed,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )

    bench.run()
