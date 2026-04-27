from typing import Generator

import pytest
import torch

from flag_gems.utils import shape_utils

from . import attri_util as attr_utils
from . import performance_utils as utils


class IsAllTrueBenchmark(utils.Benchmark):
    """
    Benchmark class for _is_all_true operation.
    _is_all_true only accepts bool tensors and reduces over all elements.
    """

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        more_shapes_1d = [
            (1025 * 1024,),
            (1024 * 1024 * 1024,),
        ]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 21, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(0, 15, 4)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            # _is_all_true only accepts bool tensors, generate random bool tensor
            inp = torch.randint(0, 2, shape, dtype=torch.bool, device=self.device)
            yield inp,


@pytest.mark.is_all_true
def test_is_all_true():
    bench = IsAllTrueBenchmark(
        op_name="is_all_true",
        torch_op=torch._is_all_true,
        dtypes=attr_utils.BOOL_DTYPES,
    )
    bench.run()
