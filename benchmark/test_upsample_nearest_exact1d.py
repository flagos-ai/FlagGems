import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class UpsampleNearestExact1dBenchmark(utils.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(2, 3, 16), (4, 8, 64), (8, 16, 256), (16, 32, 512)]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            out_size = [shape[-1] * 2]
            yield x, out_size, None


@pytest.mark.upsample_nearest_exact1d
def test_upsample_nearest_exact1d():
    bench = UpsampleNearestExact1dBenchmark(
        op_name="_upsample_nearest_exact1d",
        torch_op=torch.ops.aten._upsample_nearest_exact1d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
