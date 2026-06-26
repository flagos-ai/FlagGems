import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.reduce_sum
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_reduce_sum_perf(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="reduce_sum",
        torch_op=torch.sum,
        dtypes=[dtype],
    )

    def custom_input_iter(cur_dtype):
        for shape in bench.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, bench.device)
            yield inp,

    bench.get_input_iter = custom_input_iter
    bench.set_gems(flag_gems.reduce_sum)
    bench.run()
