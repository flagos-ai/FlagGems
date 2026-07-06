from typing import Generator

import pytest
import torch

from . import base


class AmpForeachNonFiniteCheckAndUnscaleBenchmark(base.Benchmark):
    """
    Benchmark class for _amp_foreach_non_finite_check_and_unscale_ operation.
    This operation takes a list of tensors, a found_inf tensor, and an inv_scale
    scalar tensor, and checks for non-finite values while unscaling.
    """

    # Common sizes for gradient tensors in training scenarios covering small to medium workloads
    DEFAULT_SHAPES = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    # shapes describe the first tensor; a second tensor at half size is generated
    DEFAULT_SHAPE_DESC = "M, N"

    def set_more_shapes(self):
        # larger shapes for comprehensive benchmark level
        more_shapes_2d = [(1024, 2**i) for i in range(2, 14, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(2, 10, 4)]
        return more_shapes_2d + more_shapes_3d

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            # generate 2 tensors: one at shape, one at half size in the first dim
            second_shape = (max(1, shape[0] // 2),) + shape[1:]
            tensors = [
                torch.randn(shape, device=self.device, dtype=dtype),
                torch.randn(second_shape, device=self.device, dtype=dtype),
            ]
            # PyTorch expects inv_scale and found_inf as float32
            inv_scale = torch.tensor(2.0, device=self.device, dtype=torch.float32)
            found_inf = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            yield tensors, found_inf, inv_scale


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
def test_amp_foreach_non_finite_check_and_unscale_():
    bench = AmpForeachNonFiniteCheckAndUnscaleBenchmark(
        op_name="amp_foreach_non_finite_check_and_unscale_",
        torch_op=torch._amp_foreach_non_finite_check_and_unscale_,
        # bfloat16 is not supported by the CUDA kernel for this operator
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
