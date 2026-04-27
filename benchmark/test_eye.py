import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


def _input_fn(shape, dtype, device):
    if shape[0] >= 819200:
        # Skip large shapes for performance testing
        return None

    if isinstance(shape, int):
        yield {"n": shape, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) == 1:
        n = shape[0]
        yield {"n": n, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) == 2:
        n, m = shape
        yield {"n": n, "m": m, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) > 2:
        n, m = shape[:2]
        yield {"n": n, "m": m, "dtype": dtype, "device": device},

    if base.Config.bench_level == attrs.BenchLevel.COMPREHENSIVE:
        for i in range(8, 13):
            n = 2**i
            m = 2**i
            yield {"n": n, "m": m, "dtype": dtype, "device": device},


@pytest.mark.eye
def test_eye():
    bench = base.GenericBenchmark(op_name="eye", input_fn=_input_fn, torch_op=torch.eye)
    bench.run()
