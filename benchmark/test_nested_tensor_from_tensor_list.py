import pytest
import torch

import flag_gems

from . import base, consts


class NestedTensorFromTensorListBenchmark(base.Benchmark):
    """
    Benchmark for _nested_tensor_from_tensor_list operator.
    """

    def set_shapes(self, shape_file_path=None):
        # Component-shape lists covering small/medium/large nested tensor cases.
        # Every component shares the trailing dimension so they form a valid
        # nested tensor.
        self.shapes = [
            [(2048, 4096), (2048, 4096), (2048, 4096)],
            [(64, 512, 512), (32, 512, 512), (96, 512, 512)],
            [(1024, 1024), (2048, 1024), (512, 1024), (1536, 1024)],
        ]

    def get_input_iter(self, cur_dtype):
        for shapes in self.shapes:
            tensor_list = [
                torch.randn(shape, dtype=cur_dtype, device=self.device)
                for shape in shapes
            ]
            yield (tensor_list,)

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_tensor_from_tensor_list
@pytest.mark.parametrize(
    "dtype",
    consts.FLOAT_DTYPES,
)
def test_nested_tensor_from_tensor_list(dtype):
    bench = NestedTensorFromTensorListBenchmark(
        op_name="nested_tensor_from_tensor_list",
        torch_op=flag_gems._nested_tensor_from_tensor_list,
        dtypes=[dtype],
    )
    bench.run()
