import pytest
import torch

from . import base, consts


@pytest.mark.special_entr
def test_special_entr():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_entr",
        torch_op=torch.special.entr,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
