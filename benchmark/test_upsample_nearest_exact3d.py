import pytest
import torch

from . import base, consts


@pytest.mark.upsample_nearest_exact3d
def test_upsample_nearest_exact3d():
    class UpsampleNearestExact3dBenchmark(base.Benchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [(2, 3, 8, 16, 16), (1, 1, 4, 32, 32), (2, 3, 8, 32, 32)]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for shape in self.shapes:
                x = torch.randn(shape, dtype=cur_dtype, device=self.device)
                out_size = [shape[2] * 2, shape[3] * 2, shape[4] * 2]
                yield x, out_size, None, None, None

    bench = UpsampleNearestExact3dBenchmark(
        op_name="_upsample_nearest_exact3d",
        torch_op=torch.ops.aten._upsample_nearest_exact3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
