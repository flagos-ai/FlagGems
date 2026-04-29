import pytest
import torch

import flag_gems

from . import base, consts

vendor_name = flag_gems.vendor_name


# TODO(Qiming): Kill this class
class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


def _input_fn(shape, dtype, device):
    batch, channel, height, weight = shape
    input = torch.randn(size=shape, device=device, dtype=dtype)
    scale_factors = (2, 2)
    output_size = (
        int(height * scale_factors[0]),
        int(weight * scale_factors[1]),
    )
    yield {
        "input": input,
        "output_size": output_size,
        "align_corners": False,
        "scales_h": None,
        "scales_w": None,
    },


@pytest.mark.upsample_bicubic2d_aa
def test_upsample_bicubic2d_aa():
    if vendor_name == "cambricon":
        dtypes = [torch.float32]
    elif vendor_name == "kunlunxin":
        dtypes = [torch.float32, torch.float16]
    else:
        dtypes = consts.FLOAT_DTYPES

    bench = UpsampleBenchmark(
        input_fn=_input_fn,
        op_name="upsample_bicubic2d_aa",
        torch_op=torch._C._nn._upsample_bicubic2d_aa,
        dtypes=dtypes,
    )
    bench.run()


class UpsampleBicubic2dAaBackwardBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cfgs = [
            # Small / medium — fused path targets
            (4, 16, 4, 4, 1, 1, False, "tiny 4x down"),
            (4, 16, 4, 4, 16, 16, False, "small 4x up"),
            (4, 16, 16, 16, 4, 4, False, "small 4x down"),
            (4, 16, 16, 32, 64, 128, False, "small->med 4x up"),
            (1, 1, 64, 64, 16, 16, False, "C=1 4x down"),
            (1, 1, 64, 64, 32, 32, False, "C=1 2x down"),
            (1, 1, 64, 64, 128, 128, False, "C=1 2x up"),
            (4, 3, 256, 256, 128, 128, False, "C=3 2x down"),
            (4, 3, 128, 128, 256, 256, False, "C=3 2x up"),
            (4, 64, 64, 64, 32, 32, False, "C=64 2x down"),
            # Large — 2-pass path targets
            (1, 64, 512, 512, 128, 128, False, "C=64 4x down"),
            (1, 64, 512, 512, 1024, 1024, False, "C=64 2x up"),
            (512, 1024, 32, 32, 8, 8, False, "NC=524K 4x down"),
            (256, 512, 64, 64, 16, 16, False, "NC=131K 4x down"),
            (256, 512, 64, 64, 32, 32, False, "NC=131K 2x down"),
            (256, 512, 64, 64, 128, 128, False, "NC=131K 2x up"),
        ]

    def get_input_iter(self, dtype):
        for N, C, Hi, Wi, Ho, Wo, ac, label in self._cfgs:
            grad = torch.randn([N, C, Ho, Wo], device=self.device, dtype=dtype)
            yield grad, [Ho, Wo], [N, C, Hi, Wi], ac, None, None, label

    def get_tflops(self, op, *args, **kwargs):
        grad = args[0]
        return grad.numel() * 2


@pytest.mark.upsample_bicubic2d_aa_backward
def test_upsample_bicubic2d_aa_backward():
    bench = UpsampleBicubic2dAaBackwardBenchmark(
        op_name="upsample_bicubic2d_aa_backward",
        torch_op=torch.ops.aten._upsample_bicubic2d_aa_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
