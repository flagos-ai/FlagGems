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


@pytest.mark.diff
def test_diff():
    bench = DiffBenchmark(
        op_name="diff",
        torch_op=torch.diff,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
