import pytest
import torch

from . import base, consts


class Ge_Benchmark(base.GenericBenchmarkExcluse1D):
    # Hardcoded shapes for inplace comparison: exclude 1D shapes
    # as inplace requires non-broadcastable dimensions
    DEFAULT_SHAPES = [
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
        (1024, 1024, 1024),
    ]

    def set_shapes(self, shape_file_path):
        super().set_shapes(shape_file_path)
        # Filter out 1D shapes since inplace op requires matching dimensions
        self.shapes = [s for s in self.shapes if len(s) != 1]


def ge__input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    yield inp1, inp2


def ge__torch_op(inp1, inp2):
    return inp1.clone().ge_(inp2)


@pytest.mark.ge_
def test_ge_():
    bench = Ge_Benchmark(
        op_name="ge_",
        input_fn=ge__input_fn,
        torch_op=ge__torch_op,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
