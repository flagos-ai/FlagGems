import pytest
import torch

import flag_gems

from . import base, consts


class GluJvpBenchmark(base.Benchmark):
    # glu_jvp needs glu = F.glu(x, dim), so the narrowed dim extent must be
    # even; shapes keep the same even-tail pattern as the glu benchmark.
    def set_shapes(self, shape_file_path=None):
        # Sweep covers the repo's legacy pointwise set so results stay
        # comparable with the sibling glu/glu_backward benchmarks.
        self.shapes = [
            (1024 * 1024 * 1024 // 64, 64),
            (64, 64),
            (4096, 4096),
            (64, 512, 512),
            (1024, 1024, 1024),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            dx = torch.randn(shape, dtype=cur_dtype, device=self.device)
            glu = torch.nn.functional.glu(x, dim=-1)
            yield glu, x, dx, -1


@pytest.mark.glu_jvp
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_glu_jvp(dtype):
    bench = GluJvpBenchmark(
        op_name="glu_jvp",
        torch_op=torch.ops.aten.glu_jvp,
        gems_op=flag_gems.ops.glu_jvp,
        dtypes=[dtype],
    )
    bench.run()
