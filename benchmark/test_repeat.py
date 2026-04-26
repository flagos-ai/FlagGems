import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


class RepeatBenchmark(GenericBenchmark):
    """
    RepeatBenchmark designed to evaluate tensor repeat operations along specified dimensions.
    This includes operations like tile, repeat, and repeat_interval.
    Due to potential memory limitations, benchmark sizes need to be carefully controlled.

    Notably, when the input size is set to (1024, 1024, 1024) and the repeat dimensions
    are set to [1, 1, 2], the system encountered an "illegal memory access" error.
    To avoid such issues, we constrain the benchmark input sizes for these operations
    to prevent excessive memory usage.
    """

    def set_more_shapes(self):
        return [(16, 256, 256), (512, 512, 512), (64, 64, 64, 64)]


def _input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = [1] * len(shape)
    inp2[0] = 2

    yield inp1, inp2,


@pytest.mark.repeat
def test_repeat():
    bench = RepeatBenchmark(
        op_name="repeat",
        input_fn=_input_fn,
        torch_op=torch.Tensor.repeat,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
