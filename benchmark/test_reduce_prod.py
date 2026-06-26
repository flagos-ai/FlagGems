import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.reduce_prod
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_reduce_prod_perf(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="reduce_prod",
        torch_op=torch.prod,
        dtypes=[dtype],
    )
    # reduce_prod only takes the tensor (no dim argument)
    def custom_input_iter(cur_dtype):
        for shape in bench.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, bench.device)
            yield inp,
    bench.get_input_iter = custom_input_iter
    bench.set_gems(flag_gems.reduce_prod)
    bench.run()
