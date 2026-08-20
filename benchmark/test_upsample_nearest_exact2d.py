import pytest
import torch

from . import base, consts


@pytest.mark.upsample_nearest_exact2d
def test_upsample_nearest_exact2d():
    class UpsampleNearestExact2dBenchmark(base.Benchmark):
        def set_shapes(self, shape_file_path=None):
            # Core benchmark shapes mirror the generated worktree coverage.
            self.shapes = [(2, 3, 16, 16), (4, 8, 32, 32), (8, 16, 64, 64)]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for shape in self.shapes:
                x = torch.randn(shape, dtype=cur_dtype, device=self.device)
                out_size = [shape[2] * 2, shape[3] * 2]
                yield x, out_size, None, None

    bench = UpsampleNearestExact2dBenchmark(
        op_name="upsample_nearest_exact2d",
        torch_op=torch.ops.aten._upsample_nearest_exact2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest_exact2d
def test_upsample_nearest_exact2d_out():
    op_label = "_upsample_nearest_exact2d_out"
    assert op_label

    class UpsampleNearestExact2dOutBenchmark(base.Benchmark):
        def set_shapes(self, shape_file_path=None):
            # Core benchmark shapes mirror the generated worktree coverage.
            self.shapes = [(2, 3, 16, 16), (4, 8, 32, 32), (8, 16, 64, 64)]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for shape in self.shapes:
                x = torch.randn(shape, dtype=cur_dtype, device=self.device)
                out_size = [shape[2] * 2, shape[3] * 2]
                out = torch.empty(
                    (shape[0], shape[1], out_size[0], out_size[1]),
                    dtype=cur_dtype,
                    device=self.device,
                )
                yield x, out_size, None, None, {"out": out}

    bench = UpsampleNearestExact2dOutBenchmark(
        op_name="upsample_nearest_exact2d",
        torch_op=torch.ops.aten._upsample_nearest_exact2d.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
