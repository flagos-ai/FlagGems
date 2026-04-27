from typing import Generator

import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


class CopyInplaceBenchmark(base.Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            dst = base.generate_tensor_input(shape, cur_dtype, self.device)
            src = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield dst, src


@pytest.mark.copy_
@pytest.mark.skipif(
    base.SkipVersion("torch", "<2.4"),
    reason="The copy operator requires torch >= 2.4",
)
def test_copy_inplace():
    bench = CopyInplaceBenchmark(
        op_name="copy_",
        torch_op=torch.ops.aten.copy_,
        dtypes=attrs.FLOAT_DTYPES + attrs.INT_DTYPES + attrs.BOOL_DTYPES,
        is_inplace=True,
    )

    bench.run()
