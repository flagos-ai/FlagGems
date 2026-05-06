import pytest
import torch

from . import base, consts


class UpsampleBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfgs = [
            (1, 3, 512, 512, 1024, 1024, "contiguous", None, None, "x2 nchw"),
            (8, 16, 128, 128, 256, 256, "contiguous", None, None, "x2 nchw C16"),
            (2, 3, 1024, 1024, 2048, 2048, "contiguous", None, None, "x2 nchw big"),
            (16, 16, 512, 512, 1024, 1024, "contiguous", None, None, "x2 nchw large"),
            (16, 16, 1024, 1024, 2048, 2048, "contiguous", None, None, "x2 nchw huge"),
            (1, 3, 127, 127, 255, 255, "contiguous", None, None, "near2 nchw"),
            (1, 16, 64, 64, 135, 147, "contiguous", 2.1, 2.3, "explicit nchw"),
            (1, 64, 64, 64, 135, 147, "channels_last", 2.1, 2.3, "explicit nhwc C64"),
            (1, 128, 64, 64, 135, 147, "channels_last", 2.1, 2.3, "explicit nhwc C128"),
            (1, 17, 64, 64, 129, 131, "channels_last", None, None, "near2 nhwc C17"),
        ]

    def get_input_iter(self, dtype):
        for N, C, Hi, Wi, Ho, Wo, layout, scales_h, scales_w, label in self._cfgs:
            input = torch.randn([N, C, Hi, Wi], device=self.device, dtype=dtype)
            if layout == "channels_last":
                input = input.contiguous(memory_format=torch.channels_last)
            yield {
                "input": input,
                "output_size": [Ho, Wo],
                "scales_h": scales_h,
                "scales_w": scales_w,
            }, label


@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d():
    bench = UpsampleBenchmark(
        op_name="upsample_nearest2d",
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
