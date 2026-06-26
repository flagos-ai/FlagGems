import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.reduce_variance
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_reduce_variance_perf(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="reduce_variance",
        torch_op=torch.var,
        dtypes=[dtype],
    )
    def custom_input_iter(cur_dtype):
        for shape in bench.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, bench.device)
            if inp.ndim > 1:
                yield inp, 0
            else:
                yield inp,
    bench.get_input_iter = custom_input_iter
    bench.set_gems(flag_gems.reduce_variance)
    bench.run()
