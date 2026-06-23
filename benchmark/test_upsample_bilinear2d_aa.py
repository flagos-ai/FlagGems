import pytest
import torch

import flag_gems

from . import base, consts

vendor_name = flag_gems.vendor_name


class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


@pytest.mark.upsample_bilinear2d_aa
def test_upsample_bilinear2d_aa():
    def upsample_bilinear2d_aa_input_fn(shape, dtype, device):
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

    if vendor_name == "cambricon":
        # 寒武纪仅支持 float32
        dtypes = [torch.float32]
    elif vendor_name == "kunlunxin":
        # 昆仑芯不支持 bfloat16
        dtypes = [torch.float32, torch.float16]
    else:
        dtypes = consts.FLOAT_DTYPES
    bench = UpsampleBenchmark(
        input_fn=upsample_bilinear2d_aa_input_fn,
        op_name="upsample_bilinear2d_aa",
        torch_op=torch._C._nn._upsample_bilinear2d_aa,
        dtypes=dtypes,
    )
    bench.run()
