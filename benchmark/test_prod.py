import pytest
import torch

from . import base, consts


class ProdBenchmark(base.UnaryReductionBenchmark):
    """Benchmark for prod without dim (1D shapes only)."""

    def set_more_shapes(self):
        return [s for s in super().set_more_shapes() if len(s) == 1]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [s for s in self.shapes if len(s) == 1]


class ProdDimIntBenchmark(base.UnaryReductionBenchmark):
    """Benchmark for prod with dim (2D+ shapes only)."""

    def set_more_shapes(self):
        return [s for s in super().set_more_shapes() if len(s) > 1]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [s for s in self.shapes if len(s) > 1]


@pytest.mark.prod
def test_prod():
    bench = ProdBenchmark(
        op_name="prod", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.prod_dim_int
def test_prod_dim_int():
    bench = ProdDimIntBenchmark(
        op_name="prod_dim_int", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
