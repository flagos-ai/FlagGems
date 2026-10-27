import pytest
import torch

import flag_gems

from . import base, consts


CFGS = [
    (4, 16, 4, 4, 1, 1, False), (4, 16, 4, 4, 16, 16, False),
    (4, 16, 16, 16, 4, 4, False), (4, 16, 16, 32, 64, 128, False),
    (1, 1, 64, 64, 16, 16, False), (1, 1, 64, 64, 32, 32, False),
    (1, 1, 64, 64, 128, 128, False), (4, 3, 256, 256, 128, 128, False),
    (4, 3, 128, 128, 256, 256, False), (4, 64, 64, 64, 32, 32, False),
    (1, 64, 512, 512, 128, 128, False), (1, 64, 512, 512, 1024, 1024, False),
    (512, 1024, 32, 32, 8, 8, False), (256, 512, 64, 64, 16, 16, False),
    (256, 512, 64, 64, 32, 32, False), (256, 512, 64, 64, 128, 128, False),
]


class UpsampleBilinear2dAaBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CFGS

    def get_case_iter(self, dtype):
        for ordinal, cfg in enumerate(self.shapes):
            N, C, Hi, Wi, Ho, Wo, ac = cfg
            yield self._case_from_plan(dtype, ordinal, base.BenchmarkCasePlan(
                shape={"grad_output": [N, C, Ho, Wo], "input_size": [N, C, Hi, Wi]},
                params={"output_size": [Ho, Wo], "align_corners": ac},
                builder_args=(cfg,),
            ))

    def build_inputs(self, case):
        if flag_gems.vendor_name == "enflame" and case.dtype in (
            torch.float16, torch.bfloat16
        ):
            pytest.skip(
                "Enflame native AA bilinear backward falls back to CPU, whose "
                "kernel does not support Half or BFloat16; float32 is supported"
            )
        N, C, Hi, Wi, Ho, Wo, ac = case.builder_args[0].builder_args[0]
        grad = torch.randn((N, C, Ho, Wo), device=self.device, dtype=case.dtype)
        return grad, [Ho, Wo], [N, C, Hi, Wi], ac, None, None

    def get_tflops(self, op, *args, **kwargs):
        return args[0].numel() * 2


@pytest.mark.upsample_bilinear2d_aa_backward
@pytest.mark.skipif(flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working")
def test_upsample_bilinear2d_aa_backward():
    UpsampleBilinear2dAaBackwardBenchmark(op_name="upsample_bilinear2d_aa_backward", torch_op=torch.ops.aten._upsample_bilinear2d_aa_backward, dtypes=consts.FLOAT_DTYPES).run()
