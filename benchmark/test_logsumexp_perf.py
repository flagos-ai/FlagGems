import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


def logsumexp_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, 1


@pytest.mark.logsumexp
def test_perf_logsumexp():
    bench = GenericBenchmark(
        input_fn=logsumexp_input_fn,
        op_name="logsumexp",
        torch_op=torch.logsumexp,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.logsumexp)
    bench.run()
