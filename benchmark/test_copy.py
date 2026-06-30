from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class CopyInplaceBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            dst = utils.generate_tensor_input(shape, dtype, self.device)
            src = utils.generate_tensor_input(shape, dtype, self.device)
            yield dst, src


@pytest.mark.copy_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace():
    bench = CopyInplaceBenchmark(
        op_name="copy_",
        torch_op=torch.ops.aten.copy_,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )

    bench.run()


class CopyFunctionalBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            template = utils.generate_tensor_input(shape, dtype, self.device)
            src = utils.generate_tensor_input(shape, dtype, self.device)
            yield template, src


@pytest.mark.copy
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_functional():
    bench = CopyFunctionalBenchmark(
        op_name="copy",
        torch_op=torch.ops.aten.copy,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )

    bench.run()
