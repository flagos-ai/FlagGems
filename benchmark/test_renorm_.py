import pytest
import torch

from . import base, consts


class RenormBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # 2D shapes: (num_slices, slice_size) to benchmark renorm across different dimensions
        self.shapes = [
            (16, 256),
            (32, 512),
            (64, 1024),
            (128, 2048),
            (256, 4096),
            (512, 8192),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, 2.0, 1, 1.0


@pytest.mark.renorm_
def test_renorm_():
    bench = RenormBenchmark(
        op_name="renorm_",
        torch_op=torch.ops.aten.renorm_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
