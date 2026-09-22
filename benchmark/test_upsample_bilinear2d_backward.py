# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import pytest
import torch
import triton

import flag_gems

from . import base

_MUSA_NATIVE_BUG = False
if flag_gems.vendor_name == "mthreads":
    import torch_musa

    _MUSA_NATIVE_BUG = torch_musa.__version__ == "2.9.1+a18d871"

pytestmark = pytest.mark.skipif(
    _MUSA_NATIVE_BUG,
    reason="TorchMUSA 2.9.1+a18d871 native backward executes nearest rather than bilinear; no valid native baseline",
)

_CANN_VERSION = None
if flag_gems.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _CANN_VERSION = get_cann_version()
_CANN_850 = _CANN_VERSION == "8.5.0"

DTYPES = [
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name in ("ascend", "mthreads", "iluvatar"),
            reason="FP64 arithmetic unavailable on Iluvatar; native backward rejects FP64 on Ascend/MUSA",
        ),
    ),
    torch.float16,
    torch.bfloat16,
]
CASES = [
    ("identity", (2, 16, 64, 64), (64, 64), None, None, "contiguous"),
    ("2x_small", (1, 1, 7, 9), (14, 18), None, None, "contiguous"),
    ("2x_medium", (2, 16, 64, 64), (128, 128), None, None, "contiguous"),
    ("2x_large", (4, 32, 128, 128), (256, 256), None, None, "contiguous"),
    ("fractional", (2, 8, 31, 47), (53, 79), None, None, "contiguous"),
    ("downsample", (2, 16, 64, 96), (32, 48), None, None, "contiguous"),
    ("strong_downsample", (2, 16, 96, 128), (7, 9), None, None, "contiguous"),
    ("out_one", (2, 8, 31, 47), (1, 1), None, None, "contiguous"),
    ("in_h_one", (2, 4, 1, 33), (7, 65), None, None, "contiguous"),
    ("explicit", (2, 8, 17, 23), (27, 37), 2.1, 1.8, "contiguous"),
    ("explicit_downsample", (2, 8, 33, 41), (9, 13), 0.4, 0.5, "contiguous"),
    ("strided", (2, 8, 31, 47), (61, 89), None, None, "strided"),
    ("channels_last", (2, 16, 31, 47), (61, 89), None, None, "channels_last"),
]


class UpsampleBilinear2dBackwardBenchmark(base.Benchmark):
    def __init__(self, case, align_corners, out_variant, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case = case
        self.align_corners = align_corners
        self.out_variant = out_variant

    def get_latency(self, op, *args, **kwargs):
        if (
            flag_gems.vendor_name != "ascend"
            or base.Config.mode != base.consts.BenchMode.KERNEL
        ):
            return super().get_latency(op, *args, **kwargs)
        if _CANN_VERSION != "9.0.0":
            raise RuntimeError(
                f"Unverified CANN {_CANN_VERSION} profiler contract; "
                "expected CANN 9.0.0 with 5 warmup and 30 active calls"
            )

        # A call can contain multiple kernels or an SDMA copy without a kernel.
        # Aggregate every device task, including copies, for the complete call.
        profile_root = Path(
            os.environ.get("FLAGGEMS_ASCEND_PROFILE_DIR", "outputs/ascend_profiles")
        )
        profile_root.mkdir(parents=True, exist_ok=True)
        implementation = "native" if op is self.torch_op else "gems"
        prefix = (
            f"{self.op_name}-{self.case[0]}-{args[0].dtype}-"
            f"align{int(self.align_corners)}-{implementation}-"
        )
        profile_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=profile_root))
        triton.backends.ascend.testing.do_bench_npu(
            lambda: op(*args, **kwargs),
            warmup=5,
            active=30,
            prof_dir=str(profile_dir),
            keep_res=True,
        )
        csv_paths = list(profile_dir.rglob("task_time_*.csv"))
        if len(csv_paths) != 1:
            raise RuntimeError(f"Expected one device task_time CSV in {profile_dir}")
        with csv_paths[0].open(newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
            rows = sorted(
                (
                    row
                    for row in raw_rows
                    if row["kernel_type"]
                    not in ("PROFILING_ENABLE", "PROFILING_DISABLE")
                ),
                key=lambda row: float(row["task_start(us)"]),
            )
        if not rows or len(rows) % 35:
            raise RuntimeError(f"Incomplete 35-call profile: {csv_paths[0]}")
        width = len(rows) // 35
        sequence = [(row["kernel_name"], row["kernel_type"]) for row in rows[:width]]
        durations = []
        for start in range(0, len(rows), width):
            group = rows[start : start + width]
            if [(row["kernel_name"], row["kernel_type"]) for row in group] != sequence:
                raise RuntimeError(f"Device task sequence changed: {csv_paths[0]}")
            task_durations = [float(row["task_time(us)"]) for row in group]
            if any(not math.isfinite(value) or value <= 0 for value in task_durations):
                raise RuntimeError(f"Invalid device task duration: {csv_paths[0]}")
            durations.append(math.fsum(task_durations))
        latency = math.fsum(durations[5:]) / 30 / 1000
        (profile_dir / "aggregation.json").write_text(
            json.dumps(
                {
                    "csv": str(csv_paths[0]),
                    "tasks_per_call": width,
                    "task_sequence": sequence,
                    "raw_rows": len(raw_rows),
                    "excluded_profiler_events": len(raw_rows) - len(rows),
                    "call_duration_us": durations,
                    "warmup_calls": 5,
                    "active_calls": 30,
                    "latency_ms": latency,
                },
                indent=2,
            )
            + "\n"
        )
        return latency

    def get_input_iter(self, dtype):
        _, input_size, output_size, sh, sw, layout = self.case
        n, c, ih, iw = input_size
        oh, ow = output_size
        grad = torch.randn(
            (n, c, oh, ow * 2 if layout == "strided" else ow),
            device=self.device,
            dtype=dtype,
        )
        if layout == "strided":
            grad = grad[..., ::2]
        if layout == "channels_last":
            grad = grad.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
        args = (grad, output_size, input_size, self.align_corners, sh, sw)
        if self.out_variant:
            output = torch.empty(
                (n, c, ih, iw * 2 if layout == "strided" else iw),
                device=self.device,
                dtype=dtype,
            )
            if layout == "strided":
                output = output[..., ::2]
            if layout == "channels_last":
                output = output.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
            yield (*args, {"grad_input": output})
        else:
            yield args


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.skipif(
    _CANN_850,
    reason="CANN 8.5.0 is accuracy-only: its dynamic profiler iteration count does not satisfy the 35-call contract",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
def test_upsample_bilinear2d_backward(case, align_corners, dtype):
    bench = UpsampleBilinear2dBackwardBenchmark(
        case,
        align_corners,
        False,
        op_name="upsample_bilinear2d_backward",
        torch_op=torch.ops.aten.upsample_bilinear2d_backward.default,
        gems_op=flag_gems.ops.upsample_bilinear2d_backward,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.skipif(
    _CANN_850,
    reason="CANN 8.5.0 is accuracy-only: its dynamic profiler iteration count does not satisfy the 35-call contract",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
def test_upsample_bilinear2d_backward_grad_input(case, align_corners, dtype):
    bench = UpsampleBilinear2dBackwardBenchmark(
        case,
        align_corners,
        True,
        op_name="upsample_bilinear2d_backward.grad_input",
        torch_op=torch.ops.aten.upsample_bilinear2d_backward.grad_input,
        gems_op=flag_gems.ops.upsample_bilinear2d_backward_grad_input,
        dtypes=[dtype],
    )
    bench.run()
