import pytest
import torch

from . import base, consts


class GluBenchmark(base.UnaryPointwiseBenchmark):
    # Glu test requires even numbers
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(1, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(1, 15, 4)]
        return special_shapes_2d + sp_shapes_3d


@pytest.mark.glu
def test_glu():
    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch.nn.functional.glu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.glu_backward
def test_glu_backward():
    bench = GluBenchmark(
        op_name="glu_backward",
        torch_op=torch.nn.functional.glu,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
