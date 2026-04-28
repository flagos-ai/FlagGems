import pytest
import torch

from . import base, consts


class UpsampleNearestExact1dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(2, 3, 16), (4, 8, 64), (8, 16, 256), (16, 32, 512)]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            out_size = [shape[-1] * 2]
            yield x, out_size, None


@pytest.mark.upsample_nearest_exact1d
def test_upsample_nearest_exact1d():
    bench = UpsampleNearestExact1dBenchmark(
        op_name="_upsample_nearest_exact1d",
        torch_op=torch.ops.aten._upsample_nearest_exact1d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
