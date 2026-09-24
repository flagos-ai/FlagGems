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

_MUSA_NATIVE_IS_NEAREST = False
if flag_gems.vendor_name == "mthreads":
    import torch_musa

    _MUSA_NATIVE_IS_NEAREST = torch_musa.__version__ == "2.9.1+a18d871"
_INVALID_MUSA_NATIVE = pytest.mark.skipif(
    _MUSA_NATIVE_IS_NEAREST,
    reason=(
        "TorchMUSA 2.9.1+a18d871 native bicubic backward executes nearest; "
        "no valid bicubic performance baseline"
    ),
)

_CANN_VERSION = None
if flag_gems.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _CANN_VERSION = get_cann_version()

pytestmark = pytest.mark.skipif(
    _CANN_VERSION == "8.5.0",
    reason=(
        "CANN 8.5.0 is accuracy-only for this operator; its dynamic profiler "
        "iterations do not satisfy the CANN 9.0 35-call timing contract"
    ),
)

DTYPES = [
    pytest.param(torch.float32, marks=_INVALID_MUSA_NATIVE),
    pytest.param(
        torch.float64,
        marks=[
            pytest.mark.skipif(
                flag_gems.vendor_name in ("mthreads", "ascend"),
                reason="MUSA muDNN and Ascend aclnn bicubic backward reject DOUBLE; no native FP64 latency baseline",
            ),
            pytest.mark.skipif(
                flag_gems.vendor_name == "iluvatar",
                reason=(
                    "Iluvatar native FP64 bicubic backward returns zeros for "
                    "valid nonzero inputs; no valid native baseline"
                ),
            ),
        ],
    ),
    pytest.param(torch.float16, marks=_INVALID_MUSA_NATIVE),
    pytest.param(torch.bfloat16, marks=_INVALID_MUSA_NATIVE),
]
# This fixed manifest is shared by both overloads and all measured backends.
CASES = [
    ("identity_small", (1, 3, 8, 9), (8, 9), None, None, "contiguous"),
    ("identity_large", (8, 16, 128, 128), (128, 128), None, None, "contiguous"),
    ("double_small", (1, 3, 8, 9), (16, 18), None, None, "contiguous"),
    ("double_large", (4, 16, 64, 64), (128, 128), None, None, "contiguous"),
    ("noninteger", (2, 8, 37, 53), (61, 89), None, None, "contiguous"),
    ("downsample", (2, 8, 64, 80), (23, 31), None, None, "contiguous"),
    ("strong_downsample", (2, 8, 128, 160), (3, 5), None, None, "contiguous"),
    ("out_one", (2, 3, 13, 17), (1, 1), None, None, "contiguous"),
    ("input_one", (1, 3, 1, 1), (31, 37), None, None, "contiguous"),
    ("explicit_scales", (2, 3, 13, 17), (23, 31), 1.7, 1.9, "contiguous"),
    ("small_scale", (1, 3, 13, 17), (23, 31), 0.6, 0.8, "contiguous"),
    ("noncontiguous", (2, 3, 13, 17), (23, 31), None, None, "strided"),
]


class UpsampleBicubic2dBackwardBenchmark(base.Benchmark):
    case = None
    align_corners = False
    out_variant = False

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
            duration = math.fsum(task_durations)
            durations.append(duration)
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

    def set_shapes(self, shape_file_path=None):
        self.shapes = [self.case]

    def get_input_iter(self, cur_dtype):
        _, shape, output_size, sh, sw, layout = self.case
        n, c = shape[:2]
        oh, ow = output_size
        if layout == "strided":
            grad_output = torch.randn(
                (n, c, oh, ow * 2), device=self.device, dtype=cur_dtype
            )[..., ::2]
        else:
            grad_output = torch.randn(
                (n, c, oh, ow), device=self.device, dtype=cur_dtype
            )
        args = (grad_output, output_size, shape, self.align_corners, sh, sw)
        if self.out_variant:
            if layout == "strided":
                out = torch.empty(
                    (*shape[:-1], shape[-1] * 2), device=self.device, dtype=cur_dtype
                )[..., ::2]
            else:
                out = torch.empty(shape, device=self.device, dtype=cur_dtype)
            yield (*args, {"grad_input": out})
        else:
            yield args


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
def test_upsample_bicubic2d_backward(case, align_corners, dtype):
    bench = UpsampleBicubic2dBackwardBenchmark(
        op_name="upsample_bicubic2d_backward",
        torch_op=torch.ops.aten.upsample_bicubic2d_backward.default,
        gems_op=flag_gems.ops.upsample_bicubic2d_backward,
        dtypes=[dtype],
        case=case,
        align_corners=align_corners,
    )
    bench.run()


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
@pytest.mark.upsample_bicubic2d_backward_grad_input
def test_upsample_bicubic2d_backward_grad_input(case, align_corners, dtype):
    bench = UpsampleBicubic2dBackwardBenchmark(
        op_name="upsample_bicubic2d_backward.grad_input",
        torch_op=torch.ops.aten.upsample_bicubic2d_backward.grad_input,
        gems_op=flag_gems.ops.upsample_bicubic2d_backward_grad_input,
        dtypes=[dtype],
        case=case,
        align_corners=align_corners,
        out_variant=True,
    )
    bench.run()
