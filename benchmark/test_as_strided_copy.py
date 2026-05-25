from typing import Generator

import pytest
import torch

from . import base, consts, utils

AS_STRIDED_COPY_SHAPES = [
    ((64, 64), (64, 64), (64, 1), 0),
    ((256, 256), (128, 128), (1, 256), 0),
    ((1024, 1024), (512, 512), (1024, 1), 0),
    ((64, 128, 64), (32, 64, 32), (8192, 64, 2), 0),
    ((1024 * 1024,), (512 * 1024,), (2,), 0),
]


class AsStridedCopyBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, size, stride, storage_offset"

    def set_shapes(self, shape_file_path=None):
        self.shapes = AS_STRIDED_COPY_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for input_shape, size, stride, storage_offset in self.shapes:
            inp = utils.generate_tensor_input(input_shape, dtype, self.device)
            yield inp, size, stride, storage_offset


class AsStridedCopyOutBenchmark(AsStridedCopyBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for input_shape, size, stride, storage_offset in self.shapes:
            inp = utils.generate_tensor_input(input_shape, dtype, self.device)
            out = torch.empty(size, dtype=dtype, device=self.device)
            yield inp, size, stride, storage_offset, {"out": out}


@pytest.mark.as_strided_copy
def test_as_strided_copy():
    bench = AsStridedCopyBenchmark(
        op_name="as_strided_copy",
        torch_op=torch.ops.aten.as_strided_copy,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.as_strided_copy_out
def test_as_strided_copy_out():
    bench = AsStridedCopyOutBenchmark(
        op_name="as_strided_copy_out",
        torch_op=torch.ops.aten.as_strided_copy,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
