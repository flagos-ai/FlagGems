import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.as_strided_
def test_as_strided_():
    def as_strided_input_fn(shape, dtype, device):
        # Generate input tensor
        inp = torch.randn(shape, dtype=dtype, device=device)
        # Define different size/stride combinations to test
        size_stride_combos = [
            ([shape[0] // 2, 2], [2, 1]),
            ([2, shape[0] // 2], [shape[0] // 2, 1]),
        ]
        for size, stride in size_stride_combos:
            # Skip if size product doesn't match
            if size[0] * size[1] != shape[0]:
                continue
            yield inp, size, stride

    bench = base.GenericBenchmark(
        input_fn=as_strided_input_fn,
        op_name="as_strided_",
        torch_op=torch.as_strided_,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.set_gems(flag_gems.as_strided_)
    bench.run()
