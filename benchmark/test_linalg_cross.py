from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


# Mirror the layouts covered by tests/test_linalg_cross.py at the smallest
# performance scale that clears fixed kernel-launch overhead on each backend.
LINALG_CROSS_COMMON_CASES = [
    ((262144, 3, 4), (1, 3, 4), 1),
    ((1, 3), (131072, 3), -1),
    ((65536, 4, 3), (65536, 4, 3), -1),
    ((131072, 3), (131072, 3), -1),
    ((262144, 3), (262144, 3), -1),
    ((524288, 3), (524288, 3), -1),
    ((1048576, 3), (1048576, 3), -1),
]


def _randn(shape, dtype, device):
    if dtype.is_complex and flag_gems.vendor_name == "ascend":
        # torch_npu cannot generate complex normal values directly on the NPU.
        return torch.randn(shape, dtype=dtype).to(device)
    return torch.randn(shape, dtype=dtype, device=device)


class LinalgCrossBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(LINALG_CROSS_COMMON_CASES)
        if flag_gems.vendor_name == "ascend":
            self.shapes.insert(0, ((16384, 3, 4), (1, 3, 4), 1))

    def get_input_iter(self, cur_dtype) -> Generator:
        for input_shape, other_shape, dim in self.shapes:
            input = _randn(input_shape, cur_dtype, self.device)
            other = _randn(other_shape, cur_dtype, self.device)
            yield input, other, {"dim": dim}


@pytest.mark.linalg_cross
def test_linalg_cross():
    bench = LinalgCrossBenchmark(
        op_name="linalg_cross",
        torch_op=torch.linalg.cross,
        gems_op=flag_gems.linalg_cross if flag_gems.vendor_name == "ascend" else None,
        # These are the common dtypes supported by both NVIDIA and Ascend.
        dtypes=[torch.float32] + consts.COMPLEX_DTYPES,
    )
    bench.run()
