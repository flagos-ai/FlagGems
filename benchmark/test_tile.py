import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


class TileBenchmark(GenericBenchmark):
    """
    TileBenchmark designed to evaluate tensor repeat operations along specified dimensions.
    Due to potential memory limitations, benchmark sizes need to be carefully controlled.

    Notably, when the input size is set to (1024, 1024, 1024) and the repeat dimensions
    are set to [1, 1, 2], the system encountered an "illegal memory access" error.
    To avoid such issues, we constrain the benchmark input sizes for these operations
    to prevent excessive memory usage.
    """

    def set_more_shapes(self):
        more_shapes = [
            (16, 256, 256),
            (512, 512, 512),
            (64, 64, 64, 64),
        ]
        return more_shapes


def _input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    dim = [1] * len(shape)
    dim[0] = 2
    yield inp, {"dims": dim}


@pytest.mark.tile
def test_tile():
    bench = TileBenchmark(
        op_name="tile",
        input_fn=_input_fn,
        torch_op=torch.tile,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
