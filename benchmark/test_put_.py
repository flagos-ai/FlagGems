import math

import pytest
import torch

from . import base, consts

# Cover 1-D, multi-D and large flatten spans. Mirrors `test_put.py`.
PUT_BENCH_SHAPES = [
    (16,),
    (64, 64),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# `put_` is in-place so it only allocates `source` on top of `self`, but the
# default core shapes still include 1G-element tensors (~4 GiB each in fp32);
# cap the element count at 2**28 (256 MiB -> ~1 GiB fp32) to stay clear of the
# ~16 GiB OOM seen on the CI runner.
PUT_MAX_NUMEL = 2**28


class PutBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        self.shapes = PUT_BENCH_SHAPES
        return None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [s for s in self.shapes if math.prod(s) < PUT_MAX_NUMEL]


def put_input_fn(accumulate):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        numel = inp.numel()
        # Write into roughly half of the positions, capped so the core-level
        # 1G shape does not allocate ~512M indices and dominate the run.
        count = max(1, min(numel // 2, 2**20))
        # Allow duplicate indices only when accumulating; keep them unique
        # otherwise so the overwrite path is deterministic.
        if accumulate:
            index = torch.randint(0, numel, (count,), dtype=torch.int64, device=device)
        else:
            index = torch.randperm(numel, device=device)[:count].to(torch.int64)
        source = torch.randn(count, dtype=dtype, device=device, requires_grad=False)
        yield inp, index, source, accumulate

    return inner


@pytest.mark.put_
def test_put_():
    bench = PutBenchmark(
        op_name="put_",
        torch_op=torch.ops.aten.put_,
        input_fn=put_input_fn(accumulate=False),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.put_
def test_put__accumulate():
    bench = PutBenchmark(
        op_name="put_",
        torch_op=torch.ops.aten.put_,
        input_fn=put_input_fn(accumulate=True),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
