import pytest
import torch

from . import base, consts, utils


def _prod_dim_int_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 1, False


class ProdBenchmark(base.UnaryReductionBenchmark):
    """Benchmark for prod without dim (1D shapes only)."""

    def set_more_shapes(self):
        return [s for s in super().set_more_shapes() if len(s) == 1]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [s for s in self.shapes if len(s) == 1]


class ProdDimIntBenchmark(base.GenericBenchmark):
    """Benchmark for prod.dim_int (2D+ shapes only)."""

    def set_more_shapes(self):
        more_shapes_2d = [(1024, 2**i) for i in range(0, 11, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(0, 11, 4)]
        return more_shapes_2d + more_shapes_3d


@pytest.mark.prod
def test_prod():
    bench = ProdBenchmark(
        op_name="prod", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.prod_dim_int
def test_prod_dim_int():
    bench = ProdDimIntBenchmark(
        op_name="prod_dim_int",
        input_fn=_prod_dim_int_input_fn,
        torch_op=torch.prod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
