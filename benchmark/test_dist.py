import pytest
import torch

import flag_gems

from . import base, consts, utils

# p values covering every kernel path: the p = 0 / 1 / 2 fast paths, the
# general real-p path, and the inf / -inf (max / min) paths.
P_LIST = (float("-inf"), float("inf"), 0.0, 1.0, 2.0, 6.6)


def composed_dist(x, y, p=2.0):
    # torch_npu's native dist only supports p in {0, 1, 2}; compose the other
    # norms from basic torch ops (same approach as benchmark pairwise_distance).
    if p == float("inf"):
        return torch.amax(torch.abs(x - y))
    elif p == float("-inf"):
        return torch.amin(torch.abs(x - y))
    elif p == 0.0 or p == 1.0 or p == 2.0:
        return torch.dist(x, y, p)
    else:
        diff = torch.abs(x - y)
        return torch.pow(torch.sum(torch.pow(diff, p)), 1.0 / p).to(x.dtype)


def dist_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    for p in P_LIST:
        yield inp1, inp2, {"p": p}


class DistBenchmark(base.GenericBenchmark):
    # dist flattens its inputs, so 1-D small / medium / large shapes are
    # enough: the small ones exercise the single-launch path (numel <= 16384),
    # the rest the two-stage reduction path.
    SHAPES = [
        (16,),
        (64,),
        (128,),
        (512,),
        (1024,),  # small
        (16384,),  # single-launch path upper bound
        (2**20,),  # medium
        (2**24,),
        (2**28,),  # large
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.SHAPES


@pytest.mark.dist
def test_dist():
    safe_dist = torch.dist
    if base.vendor_name == "ascend":
        safe_dist = composed_dist
    bench = DistBenchmark(
        op_name="dist",
        input_fn=dist_input_fn,
        torch_op=safe_dist,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.dist)
    bench.run()
