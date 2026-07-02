import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.reduce_max
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_reduce_max_perf(dtype):
    # Create custom benchmark that only passes the tensor (not the extra dim arg)
    bench = base.UnaryReductionBenchmark(
        op_name="reduce_max",
        torch_op=torch.max,
        dtypes=[dtype],
    )

    # Override get_input_iter to only pass tensor (not the extra int arg)
    def custom_input_iter(cur_dtype):
        for shape in bench.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, bench.device)
            yield inp,

    bench.get_input_iter = custom_input_iter
    bench.set_gems(flag_gems.reduce_max)
    bench.run()
