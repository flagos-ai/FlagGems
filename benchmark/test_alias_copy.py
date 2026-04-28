import pytest
import torch

from . import base, consts


@pytest.mark.alias_copy
def test_alias_copy():
    bench = base.UnaryPointwiseBenchmark(
        op_name="alias_copy",
        torch_op=torch.ops.aten.alias_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.alias_copy_out
def test_alias_copy_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="alias_copy_out",
        torch_op=torch.ops.aten.alias_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
