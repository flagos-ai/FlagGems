from typing import Generator

import pytest
import torch

from . import base, consts


class BucketizeBenchmark(base.Benchmark):
    """
    Benchmark class for bucketize operations.
    """

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        # Use fixed boundaries for all tests
        boundaries = torch.tensor(
            [1.0, 3.0, 5.0, 7.0, 9.0], device=self.device, dtype=cur_dtype
        )
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, boundaries

    def get_tflops(self, op, *args, **kwargs):
        # Not applicable for bucketize
        return 0


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            "bucketize",
            torch.bucketize,
            consts.FLOAT_DTYPES,
            marks=getattr(pytest.mark, "bucketize", None),
        ),
    ],
)
@pytest.mark.bucketize
def test_bucketize_perf(op_name, torch_op, dtypes):
    bench = BucketizeBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()
