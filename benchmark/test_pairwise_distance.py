import pytest
import torch

import flag_gems

from . import base, consts, utils


def composed_pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    if p == float("inf"):
        diff = torch.abs(x1 - x2 + eps)
        return torch.amax(diff, dim=-1, keepdim=keepdim)
    elif p == float("-inf"):
        diff = torch.abs(x1 - x2 + eps)
        return torch.amin(diff, dim=-1, keepdim=keepdim)
    elif p == 0.0 or p == 1.0 or p == 2.0:
        return torch.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)
    else:
        diff = torch.abs(x1 - x2 + eps)
        return torch.pow(
            torch.sum(torch.pow(diff, p), dim=-1, keepdim=keepdim), 1.0 / p
        ).to(x1.dtype)


def pairwise_distance_input_fn(shape, dtype, device):
    # x1 and x2 must be broadcastable. Use identical (M, N) shapes so that
    # the op computes one p-distance per row -> M pairs of D-dim vectors,
    # matching both torch.nn.functional.pairwise_distance and the gems kernel
    # (which does N, D = x1.shape and reduces over D).
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Arbitrary real p is supported; sweep several p values plus eps.
        for p in (float("-inf"), float("inf"), 0.0, 1.0, 2.0, 6.6):
            yield inp1, inp2, {"p": p}
    else:
        yield inp1, inp2  # default p=2.0


class PairwiseDistanceBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        # Keep the parent's large-N 2-D shapes, then add the small-N-large-D
        # regime: one program per row => few rows => SM underutilization (the
        # case a 2-D / split-K grid targets). (1, D) is equivalent to a 1-D
        # single pair here (grid = 1 program).
        shapes = super().set_more_shapes()
        shapes += [(1, 65536), (8, 65536), (64, 65536), (1, 10000000)]
        return shapes


@pytest.mark.pairwise_distance
def test_pairwise_distance():
    safe_pairwise_distance = torch.pairwise_distance
    if base.vendor_name == "ascend":
        safe_pairwise_distance = composed_pairwise_distance
    bench = PairwiseDistanceBenchmark(
        op_name="pairwise_distance",
        input_fn=pairwise_distance_input_fn,
        torch_op=safe_pairwise_distance,
        gems_op=flag_gems.pairwise_distance,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
