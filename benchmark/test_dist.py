import pytest
import torch

import flag_gems

from . import base, consts, utils

# p values covering every kernel path: the p = 0 / 1 / 2 fast paths, the
# general real-p path, and the inf / -inf (max / min) paths.
P_LIST = (float("-inf"), float("inf"), 0.0, 1.0, 2.0, 6.6)


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
    bench = DistBenchmark(
        op_name="dist",
        input_fn=dist_input_fn,
        torch_op=torch.dist,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.dist)
    bench.run()
