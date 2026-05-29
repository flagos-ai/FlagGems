import pytest
import torch

from . import base, consts


def randint_input_fn(shape, dtype, device):
    high = 100
    yield high, shape


class RandintBenchmark(base.GenericBenchmarkExcluse1D):
    # Override set_more_shapes to provide custom shapes for randint
    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]


@pytest.mark.randint
def test_randint():
    bench = RandintBenchmark(
        input_fn=randint_input_fn,
        op_name="randint",
        torch_op=torch.randint,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
