import pytest
import torch

import flag_gems

from . import base, consts


def unsqueeze_copy_input_fn(shape, dtype, device):
    inp = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    return [(inp, -1)]


class UnsqueezeCopyBenchmark(base.GenericBenchmark):

    def set_shapes(self, shape_file_path=None):
        # Only benchmark representative shapes.
        # Avoid GenericBenchmark default shapes and huge extra shapes.
        self.shapes = [
            (1024,),
            (1024, 1024),
            (4096, 4096),
            (32, 1024, 1024),
        ]

    def set_more_shapes(self):
        # Disable GenericBenchmark automatic extra shapes.
        return []


@pytest.mark.unsqueeze_copy
def test_unsqueeze_copy():

    bench = UnsqueezeCopyBenchmark(
        op_name="unsqueeze_copy",
        torch_op=torch.ops.aten.unsqueeze_copy.default,
        input_fn=unsqueeze_copy_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.set_gems(flag_gems.unsqueeze_copy)

    bench.run()
