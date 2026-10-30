import pytest
import torch

import flag_gems

from . import base


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="NVIDIA-only CUDA JIT kernel; not supported on other backends",
)
@pytest.mark.special_hermite_polynomial_he
def test_special_hermite_polynomial_he():
    class _HermiteHeBenchmark(base.BinaryPointwiseBenchmark):
        def get_input_iter(self, dtype):
            for shape in self.shapes:
                inp1 = base.generate_tensor_input(shape, dtype, self.device)
                # n must be in [0, 10] per operator validation
                inp2 = torch.randint(0, 11, shape, device=self.device).to(dtype)
                yield inp1, inp2

    bench = _HermiteHeBenchmark(
        op_name="special_hermite_polynomial_he",
        torch_op=torch.special.hermite_polynomial_he,
        # CUDA does not support half/bfloat16 for this special function
        dtypes=[torch.float32],
    )
    bench.run()
