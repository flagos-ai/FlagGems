import pytest
import torch

from . import base, consts


SHAPES = [(2, 3, 16, 16), (4, 8, 32, 32), (8, 16, 64, 64)]


class UpsampleNearestExact2dBenchmark(base.Benchmark):
    def __init__(self, *args, with_out=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_out = with_out

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            out_size = (dims[2] * 2, dims[3] * 2)
            shape_map = {"input": list(dims), "output_size": list(out_size)}
            if self.with_out:
                shape_map["out"] = [dims[0], dims[1], *out_size]
            yield self._case_from_plan(dtype, ordinal, base.BenchmarkCasePlan(shape=shape_map, params={}, builder_args=(dims, out_size)))

    def build_inputs(self, case):
        shape, out_size = case.builder_args[0].builder_args
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        if self.with_out:
            out = torch.empty((shape[0], shape[1], *out_size), dtype=case.dtype, device=self.device)
            return x, list(out_size), None, None, {"out": out}
        return x, list(out_size), None, None


@pytest.mark.upsample_nearest_exact2d
@pytest.mark.underscore_upsample_nearest_exact2d
def test_upsample_nearest_exact2d():
    UpsampleNearestExact2dBenchmark(op_name="_upsample_nearest_exact2d", torch_op=torch.ops.aten._upsample_nearest_exact2d, dtypes=consts.FLOAT_DTYPES).run()


@pytest.mark.upsample_nearest_exact2d_out
def test_upsample_nearest_exact2d_out():
    UpsampleNearestExact2dBenchmark(
        op_name="upsample_nearest_exact2d_out",
        torch_op=torch.ops.aten._upsample_nearest_exact2d.out,
        dtypes=consts.FLOAT_DTYPES,
        with_out=True,
        is_inplace=True,
    ).run()
