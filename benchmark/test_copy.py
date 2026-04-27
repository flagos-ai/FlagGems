from typing import Generator

import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


class CopyInplaceBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            dst = utils.generate_tensor_input(shape, dtype, self.device)
            src = utils.generate_tensor_input(shape, dtype, self.device)
            yield dst, src


@pytest.mark.copy_
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.4"),
    reason="The copy operator requires torch >= 2.4",
)
def test_copy_inplace():
    bench = CopyInplaceBenchmark(
        op_name="copy_",
        torch_op=torch.ops.aten.copy_,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )

    bench.run()
