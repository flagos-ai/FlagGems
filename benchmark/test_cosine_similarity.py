import pytest
import torch

import flag_gems

from . import base, consts, utils


def cosine_similarity_input_fn(shape, dtype, device):
    # x1 and x2 use identical (M, N) shapes so the op computes one cosine
    # similarity per row -> M pairs of N-dim vectors, matching both
    # torch.cosine_similarity and the gems kernel (reduce over the last dim).
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, inp2, {"dim": 1, "eps": 1e-8}


class CosineSimilarityBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        # Keep the parent's large-N 2-D shapes, then add the small-N-large-D
        # regime: one program per row => few rows => SM underutilization.
        shapes = super().set_more_shapes()
        shapes += [(1, 65536), (8, 65536), (64, 65536), (1, 10000000)]
        return shapes


@pytest.mark.cosine_similarity
def test_cosine_similarity():
    bench = CosineSimilarityBenchmark(
        op_name="cosine_similarity",
        input_fn=cosine_similarity_input_fn,
        torch_op=torch.cosine_similarity,
        gems_op=flag_gems.cosine_similarity,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
