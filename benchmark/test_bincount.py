import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


def bincount_input_fn(shape, dtype, device):
    if shape[0] > 1_000_000:
        return

    n = shape[0]
    for num_classes in [10, 256, 4096]:
        inp = torch.randint(0, num_classes, (n,), dtype=torch.int64, device=device)

        yield inp, {}

        yield inp, {"minlength": max(512, num_classes * 2)}


@pytest.mark.bincount
def test_bincount():
    bench = utils.GenericBenchmark(
        input_fn=bincount_input_fn,
        op_name="bincount",
        torch_op=torch.bincount,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.bincount)
    bench.run()


def bincount_weighted_input_fn(shape, dtype, device):
    if shape[0] > 1_000_000:
        return

    n = shape[0]
    for num_classes in [10, 256, 4096]:
        inp = torch.randint(0, num_classes, (n,), dtype=torch.int64, device=device)
        weights = torch.randn((n,), dtype=dtype, device=device)

        yield inp, {"weights": weights}

        yield inp, {"weights": weights, "minlength": max(512, num_classes * 2)}


@pytest.mark.bincount
@pytest.mark.parametrize("dtype", attr_utils.FLOAT_DTYPES)
def test_bincount_weighted(dtype):
    bench = utils.GenericBenchmark(
        input_fn=bincount_weighted_input_fn,
        op_name=f"bincount_weighted_{str(dtype).split('.')[-1]}",
        torch_op=torch.bincount,
        dtypes=[dtype],
    )
    bench.set_gems(flag_gems.bincount)
    bench.run()
