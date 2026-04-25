import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class SortBenchmark(utils.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1, "descending": False},


@pytest.mark.sort
def test_perf_sort():
    bench = SortBenchmark(
        input_fn=_input_fn,
        op_name="sort",
        torch_op=torch.sort,
        dtypes=attr_utils.INT_DTYPES + attr_utils.FLOAT_DTYPES,
    )

    bench.run()
