import math

import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    if not isinstance(shape, tuple):
        return

    numel = math.prod(shape)
    if numel == 0:
        return

    if base.Config.bench_level != consts.BenchLevel.COMPREHENSIVE:
        num_classes_list = [16, 64]
    else:
        num_classes_list = [16, 64, 256]

    max_output_elems = 100_000_000

    for num_classes in num_classes_list:
        if numel * num_classes > max_output_elems:
            continue
        inp = torch.randint(0, num_classes, shape, device=device, dtype=torch.int64)
        inp.view(-1)[0] = num_classes - 1
        yield inp, num_classes
        yield inp, -1


@pytest.mark.one_hot
def test_perf_one_hot():
    bench = base.GenericBenchmark(
        op_name="one_hot",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.one_hot,
        dtypes=[torch.int64],
    )
    bench.run()
