import pytest
import torch

from . import base, consts


def sample_dirichlet_input_fn(shape, cur_dtype, device):
    alpha = torch.rand(shape, dtype=cur_dtype, device=device) * 5.0 + 0.5
    yield (alpha,)


class SampleDirichletBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # _sample_dirichlet runs a per-element rejection-sampling loop, so the
        # shared DEFAULT_SHAPES (up to a 2**30-element 1-D tensor) would run for
        # minutes / OOM. Use [batch, K] shapes with a modest number of
        # categories K, which is how the op is actually used.
        self.shapes = [(1024, 16), (4096, 32), (16384, 64), (65536, 128)]


@pytest.mark.sample_dirichlet
def test_sample_dirichlet():
    bench = SampleDirichletBenchmark(
        input_fn=sample_dirichlet_input_fn,
        op_name="sample_dirichlet",
        torch_op=torch._sample_dirichlet,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
