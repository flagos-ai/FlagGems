from typing import Generator

import pytest
import torch

from . import base


class SvdvalsBenchmark(base.BlasBenchmark):
    """
    benchmark for linalg.svdvals
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 4, 4),
            (1, 4, 8),
            (1, 8, 4),
            (1, 64, 64),
        ]
        self.shape_desc = "B, M, N"

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        # shapes are in format (b, m, n, k) - get m, n for matrix shapes
        for shape in self.shapes:
            if len(shape) >= 2:
                m, n = shape[1], shape[2] if len(shape) > 2 else shape[1]
                yield from self.input_fn(m, n, cur_dtype, self.device)


@pytest.mark.linalg_svdvals
def test_svdvals_benchmark():
    def svdvals_input_fn(m, n, cur_dtype, device):
        A = torch.randn([m, n], dtype=cur_dtype, device=device)
        yield A,

    bench = SvdvalsBenchmark(
        input_fn=svdvals_input_fn,
        op_name="linalg_svdvals",
        torch_op=torch.linalg.svdvals,
        # Only test float32 as torch doesn't support float16 SVD
        dtypes=[torch.float32],
    )
    bench.run()
