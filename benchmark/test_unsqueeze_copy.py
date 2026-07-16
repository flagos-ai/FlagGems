import pytest
import torch

import flag_gems

from . import base, consts


def unsqueeze_copy_input_fn(config, dtype, device):
    shape, dim = config

    inp = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    yield inp, dim


class UnsqueezeCopyBenchmark(base.Benchmark):

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # 1D
            ((1024,), 0),
            ((1024,), -1),

            # 2D
            ((1024, 1024), 0),
            ((1024, 1024), 1),
            ((1024, 1024), -1),

            ((4096, 4096), 0),
            ((4096, 4096), -1),

            # 3D
            ((32, 1024, 1024), 0),
            ((32, 1024, 1024), 1),
            ((32, 1024, 1024), -1),

            ((1024, 32, 1024), 1),

            # 4D
            ((8, 32, 128, 128), 1),
            ((8, 32, 128, 128), 3),
            ((8, 32, 128, 128), -1),
        ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from unsqueeze_copy_input_fn(
                config,
                dtype,
                self.device,
            )


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
