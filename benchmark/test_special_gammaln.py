import pytest
import torch

from . import base, consts


class SpecialGammalnOutBenchmark(base.UnaryPointwiseOutBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            out = torch.empty_like(inp)
            yield inp, {"out": out}


@pytest.mark.special_gammaln
def test_special_gammaln():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_gammaln",
        torch_op=torch.special.gammaln,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_gammaln_out
def test_special_gammaln_out():
    bench = SpecialGammalnOutBenchmark(
        op_name="special_gammaln_out",
        torch_op=torch.special.gammaln,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
