import pytest
import torch

from . import base, utils

# The default benchmark shapes are not suitable for linalg_vander: tiny vectors
# are dominated by launch overhead, while generic large 2D shapes can allocate
# excessive n x n outputs. These shapes keep memory bounded and exercise the
# throughput path that this kernel is intended to accelerate.
LINALG_VANDER_SHAPES = [
    (4096,),
    (8192,),
    (16384,),
    (4, 2048),
    (2, 4096),
]


class LinalgVanderBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_VANDER_SHAPES
        self.shape_desc = "input shape"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield utils.generate_tensor_input(shape, cur_dtype, self.device),


@pytest.mark.linalg_vander
def test_linalg_vander():
    bench = LinalgVanderBenchmark(
        op_name="linalg_vander",
        torch_op=torch.linalg.vander,
        # torch.linalg.vander does not support fp16/bf16 on CUDA in this PyTorch build.
        dtypes=[torch.float32],
    )
    bench.run()
