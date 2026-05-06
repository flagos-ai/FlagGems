import pytest
import torch

from . import base, consts


class UpsampleNearest2dBackwardBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfgs = [
            (4, 16, 64, 64, 128, 128, "contiguous", None, None, "x2 nchw"),
            (1, 3, 127, 127, 255, 255, "contiguous", None, None, "non-integer"),
            (4, 16, 64, 64, 256, 256, "contiguous", None, None, "x4 nchw"),
            (4, 32, 256, 256, 128, 128, "contiguous", None, None, "downsample x2"),
            (4, 32, 128, 128, 64, 64, "contiguous", None, None, "downsample"),
            (1, 8, 64, 64, 128, 192, "contiguous", None, None, "asymmetric"),
            (4, 16, 64, 64, 128, 128, "channels_last", None, None, "x2 nhwc"),
            (1, 64, 64, 64, 128, 128, "channels_last", None, None, "x2 nhwc C64"),
            (1, 128, 64, 64, 135, 147, "channels_last", 2.1, 2.3, "generic nhwc C128"),
            (1, 3, 64, 64, 135, 147, "contiguous", 2.1, 2.3, "explicit scales"),
        ]

    def get_input_iter(self, dtype):
        for N, C, Hi, Wi, Ho, Wo, layout, scales_h, scales_w, label in self._cfgs:
            grad = torch.randn([N, C, Ho, Wo], device=self.device, dtype=dtype)
            if layout == "channels_last":
                grad = grad.contiguous(memory_format=torch.channels_last)
            yield grad, [Ho, Wo], [N, C, Hi, Wi], scales_h, scales_w, label

    def get_tflops(self, op, *args, **kwargs):
        grad = args[0]
        return grad.numel()


@pytest.mark.upsample_nearest2d_backward
def test_upsample_nearest2d_backward():
    bench = UpsampleNearest2dBackwardBenchmark(
        op_name="upsample_nearest2d_backward",
        torch_op=torch.ops.aten.upsample_nearest2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
