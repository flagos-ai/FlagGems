import pytest
import torch

from . import base

pytestmark = pytest.mark.filterwarnings(
    "ignore:Warning only once for all operators.*:UserWarning"
)


class SvdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "(*B), M, N"
    DEFAULT_DTYPES = [torch.float32]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            yield inp, {"some": True, "compute_uv": True}


@pytest.mark.svd
def test_svd():
    bench = SvdBenchmark(op_name="svd", torch_op=torch.svd)
    bench.run()
