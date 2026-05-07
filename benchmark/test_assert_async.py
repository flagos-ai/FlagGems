import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    if dtype == torch.bool:
        tensor = torch.ones(shape, dtype=dtype, device=device)
    else:
        tensor = torch.ones(shape, dtype=dtype, device=device)

    msg = "Benchmark assert_async"

    yield (
        tensor,
        msg,
    )


class AssertAsyncBenchmark(base.GenericBenchmark):
    # TODO(Qiming): Is this necessary?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (),
            (1,),
            (1, 1),
            (1, 1, 1),
        ]

    def set_more_shapes(self):
        return None


@pytest.mark.assert_async
def test_assert_async():
    bench = AssertAsyncBenchmark(
        op_name="assert_async",
        input_fn=_input_fn,
        torch_op=torch._assert_async,
        dtypes=[
            torch.bool,
            torch.int32,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )

    bench.set_gems(flag_gems._assert_async)
    bench.run()
