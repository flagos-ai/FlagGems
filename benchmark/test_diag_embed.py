import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class EmbeddingBenchmark(utils.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return None


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield {"input": inp},

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        yield {"input": inp, "offset": 1, "dim1": 0, "dim2": -1},


@pytest.mark.diag_embed
def test_diag_embed():
    bench = EmbeddingBenchmark(
        op_name="diag_embed",
        input_fn=_input_fn,
        torch_op=torch.diag_embed,
        dtypes=attr_utils.FLOAT_DTYPES + attr_utils.INT_DTYPES + attr_utils.BOOL_DTYPES,
    )

    bench.run()
