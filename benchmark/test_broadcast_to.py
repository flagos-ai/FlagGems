import torch

from . import base, consts


class BroadcastToBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((1024,), (1, 1024)),          # 1D -> 2D broadcast (add leading dim)
            ((64, 1), (64, 4096)),         # 2D -> 2D broadcast (expand dim 1)
            ((1, 64), (4096, 64)),         # 2D -> 2D broadcast (expand dim 0)
            ((1, 1, 1), (64, 512, 512)),   # 3D -> 3D broadcast (expand all dims)
        ]
        self.shape_desc = "src_shape -> target_shape"

    def get_input_iter(self, dtype):
        for src_shape, target_shape in self.shapes:
            x = base.generate_tensor_input(src_shape, dtype, self.device)
            yield (x, target_shape)


def test_broadcast_to():
    benchmark = BroadcastToBenchmark(
        op_name="broadcast_to",
        torch_op=torch.broadcast_to,
        dtypes=consts.FLOAT_DTYPES,
    )
    benchmark.run()
