import math

import pytest
import torch

import flag_gems

from . import base, consts, utils


class DistBenchmark(base.GenericBenchmark):
    MAX_ELEMENTS = 2**28

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if flag_gems.vendor_name == "metax":
            self.shapes = [
                shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
            ]


@pytest.mark.dist
def test_dist():
    bench = DistBenchmark(
        op_name="dist",
        input_fn=utils.binary_input_fn,
        torch_op=torch.dist,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
