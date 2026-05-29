import functools

import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield (inp,)


class DiffBenchmark(base.GenericBenchmark2DOnly):
    def set_shapes(self, *args, **kwargs):
        super().set_shapes(*args, **kwargs)
        self.shapes = [s for s in self.shapes if all(d >= 2 for d in s)]
        self.shapes += [
            (16,),
            (4096,),
            (64, 128, 256),
            (32, 1024, 1024),
            (8, 4096, 4096),
        ]


@pytest.mark.diff
def test_diff():
    bench = DiffBenchmark(
        op_name="diff",
        torch_op=torch.diff,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.diff
def test_diff_n2():
    bench = DiffBenchmark(
        op_name="diff_n2",
        torch_op=functools.partial(torch.diff, n=2),
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
