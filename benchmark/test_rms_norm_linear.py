import pytest
import torch

import flag_gems

from . import base

# shapes for fused rms_norm_linear perf test, covering typical (batch, hidden, output_features) combinations
RMSNORM_LINEAR_PERF_SHAPES = [
    (32, 64, 128),  # (batch, hidden, output_features)
    (64, 128, 256),
    (128, 256, 512),
    (256, 512, 1024),
]


class RmsNormLinearBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RMSNORM_LINEAR_PERF_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            B, N, out_dim = shape
            x = torch.randn(B, N, dtype=cur_dtype, device=self.device)
            rms_weight = torch.randn(N, dtype=cur_dtype, device=self.device)
            linear_weight = torch.randn(out_dim, N, dtype=cur_dtype, device=self.device)
            linear_bias = torch.randn(out_dim, dtype=cur_dtype, device=self.device)
            yield x, [N], rms_weight, linear_weight, linear_bias


@pytest.mark.rms_norm_linear
def test_rms_norm_linear():
    # float32 excluded: kernel optimized for fp16/bf16, fp32 not used in typical inference scenarios
    bench = RmsNormLinearBenchmark(
        op_name="rms_norm_linear",
        torch_op=flag_gems.ops.rms_norm_linear,
        # dtypes subset of consts.FLOAT_DTYPES, float32 excluded as kernel targets fp16/bf16
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
