import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import Benchmark, generate_tensor_input


class GluBenchmark(Benchmark):
    DEFAULT_METRICS = Benchmark.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(1, 20, 4)]
        special_shapes_3d = [(64, 64, 2**i) for i in range(1, 15, 4)]
        return special_shapes_2d + special_shapes_3d

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            if shape[-1] % 2 == 0:
                yield generate_tensor_input(shape, dtype, self.device),

    def get_tflops(self, op, *args, **kwargs):
        return torch.tensor(args[0].shape).prod().item()


@pytest.mark.glu
def test_glu():
    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch.nn.functional.glu,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.glu_backward
def test_glu_backward():
    bench = GluBenchmark(
        op_name="glu_backward",
        torch_op=torch.nn.functional.glu,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
