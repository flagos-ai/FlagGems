import pytest
import torch

import flag_gems

from . import base, consts

ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS_COMPREHENSIVE = (
    # Identity
    ((2, 256, 2, 14, 14), (2, 14, 14)),
    # Global pool
    ((1, 256, 1, 64, 64), (1, 1, 1)),
    ((1, 4, 8, 24, 32), (1, 1, 1)),
    # Path B (out_d=1)
    ((1, 8, 64, 256, 256), (1, 7, 7)),
    ((8, 64, 16, 112, 112), (1, 7, 7)),
    # B1 / B2 (HW identity)
    ((1, 256, 64, 112, 112), (1, 112, 112)),
    ((4, 128, 32, 64, 64), (8, 64, 64)),
    # Path A (in_d == out_d)
    ((2, 128, 4, 28, 28), (4, 14, 14)),
    # Path C (general)
    ((2, 640, 64, 42, 72), (16, 14, 14)),
    # Large batch / channel
    ((32, 512, 8, 8, 8), (4, 4, 4)),
    ((1, 3, 5, 13, 17), (2, 5, 7)),
    # Tiny
    ((1, 4, 8, 16, 16), (4, 8, 8)),
    # Video model shapes
    ((1, 768, 160, 32, 32), (8, 32, 32)),
    ((1, 1280, 48, 36, 50), (8, 8, 8)),
    ((1, 3, 16, 224, 224), (8, 7, 7)),
    # Path F / H / I (large window)
    ((1, 16, 64, 128, 128), (4, 8, 8)),
    ((1, 8, 64, 256, 256), (64, 7, 7)),
    ((2, 64, 64, 256, 256), (2, 32, 32)),
    # Edge cases
    ((2, 256, 16, 1, 14), (8, 1, 14)),
    ((1, 32, 64, 64, 64), (4, 4, 4)),
)


def _input_fn(shapes, dtype, device):
    """Yield (input_tensor, output_size) pair for each shape config."""
    input_shape, output_size = shapes
    inp = base.generate_tensor_input(input_shape, dtype, device)
    yield inp, output_size


# ============================================================================
# Core benchmark shapes — one representative set covering every dispatch path
# with a few sizes each, plus real video-model shapes.  Used at --level core.
# ============================================================================
ADAPTIVE_MAX_POOL3D_BENCH_QUICK_CONFIGS = (
    ((2, 256, 2, 14, 14), (2, 14, 14)),
    ((1, 256, 1, 64, 64), (1, 1, 1)),
    ((1, 8, 64, 256, 256), (1, 7, 7)),
    ((1, 64, 8, 56, 56), (1, 56, 56)),
    ((4, 128, 32, 64, 64), (8, 64, 64)),
    ((2, 128, 4, 28, 28), (4, 14, 14)),
    ((2, 640, 64, 42, 72), (16, 14, 14)),
    ((1, 3, 5, 6, 7), (2, 3, 4)),
    ((1, 4, 8, 16, 16), (4, 8, 8)),
    ((1, 768, 160, 32, 32), (8, 32, 32)),
)


class AdaptiveMaxPool3dBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Return None to bypass base's dict.fromkeys dedup, which fails on
        # the nested [input_shape, output_size] list shapes in core_shapes.yaml.
        # (Same pattern as test_index.py.)
        return None

    def init_user_config(self):
        super().init_user_config()
        # On Ascend, default to OPERATOR timing (simple time-based, no profiler);
        # the profiler (do_bench_npu) is extremely slow and mis-times the Triton
        # kernels.  This runs at benchmark time (after pytest_configure), so
        # base.Config is already set (unlike module-import time).
        if flag_gems.vendor_name == "ascend":
            base.Config.mode = consts.BenchMode.OPERATOR
        # Always use file-defined shapes at both core and comprehensive levels,
        # overriding the YAML core_shapes.yaml subset.
        self.shapes = ADAPTIVE_MAX_POOL3D_BENCH_QUICK_CONFIGS
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = list(
                dict.fromkeys(ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS_COMPREHENSIVE)
            )


@pytest.mark.adaptive_max_pool3d
def test_perf_adaptive_max_pool3d():
    bench = AdaptiveMaxPool3dBenchmark(
        input_fn=_input_fn,
        op_name="adaptive_max_pool3d",
        torch_op=torch.nn.functional.adaptive_max_pool3d,
        gems_op=flag_gems.adaptive_max_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
