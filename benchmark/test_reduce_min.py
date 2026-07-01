import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.reduce_min
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_reduce_min_perf(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="reduce_min",
        torch_op=torch.amin,
        dtypes=[dtype],
    )

    # Override get_input_iter to only pass tensor (not extra args)
    def custom_input_iter(cur_dtype):
        for shape in bench.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, bench.device)
            yield inp,

    bench.get_input_iter = custom_input_iter
    bench.set_gems(flag_gems.reduce_min)
    bench.run()
