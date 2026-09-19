import pytest
import torch

from . import base, consts


@pytest.mark.underscore_conj_physical
def test__conj_physical():
    # _conj_physical only changes complex dtypes (for real dtypes it is a no-op),
    # so restrict the benchmark to complex dtypes.
    bench = base.UnaryPointwiseBenchmark(
        op_name="_conj_physical",
        torch_op=torch._conj_physical,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_conj_physical_out
def test__conj_physical_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="_conj_physical_out",
        torch_op=torch.ops.aten._conj_physical.out,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()
