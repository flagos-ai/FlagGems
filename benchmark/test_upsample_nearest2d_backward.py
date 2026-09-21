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

_CANN850_ACCURACY_ONLY = False
if flag_gems.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _CANN850_ACCURACY_ONLY = get_cann_version() == "8.5.0"

# Freeze this full matrix before timing: every variant/dtype runs every row.
# Fields: name, input_size, output_size, scales, grad_output layout, output layout.
CASES = [
    ("identity-small", (1, 3, 7, 9), (7, 9), (None, None), "contiguous", "contiguous"),
    (
        "identity-large",
        (8, 32, 128, 128),
        (128, 128),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    ("identity-explicit", (2, 8, 31, 47), (31, 47), (0.7, 2.2), "strided", "strided"),
    ("double-small", (2, 3, 7, 9), (14, 18), (None, None), "contiguous", "contiguous"),
    (
        "double-medium",
        (8, 16, 64, 64),
        (128, 128),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "double-large",
        (2, 32, 128, 128),
        (256, 256),
        (2.0, 2.0),
        "contiguous",
        "contiguous",
    ),
    (
        "double-channels-last",
        (4, 32, 63, 65),
        (126, 130),
        (None, None),
        "channels_last",
        "channels_last",
    ),
    (
        "fractional-small",
        (3, 5, 7, 11),
        (19, 23),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "fractional-large",
        (4, 16, 127, 129),
        (193, 257),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "downsample",
        (4, 16, 127, 129),
        (53, 71),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "strong-downsample",
        (2, 8, 127, 255),
        (3, 5),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    ("output-one", (4, 8, 31, 47), (1, 1), (None, None), "contiguous", "contiguous"),
    ("input-one", (2, 3, 1, 1), (31, 47), (None, None), "contiguous", "contiguous"),
    ("strided-output", (2, 8, 31, 47), (67, 99), (None, None), "strided", "strided"),
    (
        "explicit-fractional",
        (2, 3, 31, 47),
        (67, 99),
        (2.2, 2.125),
        "contiguous",
        "contiguous",
    ),
    (
        "explicit-below-one",
        (2, 3, 7, 11),
        (19, 23),
        (0.7, 0.7),
        "contiguous",
        "contiguous",
    ),
    (
        "explicit-double",
        (2, 3, 7, 11),
        (14, 22),
        (0.7, 2.2),
        "contiguous",
        "contiguous",
    ),
    (
        "one-sided-scale",
        (2, 3, 7, 11),
        (19, 23),
        (None, 0.4),
        "contiguous",
        "contiguous",
    ),
    (
        "one-equal-axis",
        (2, 8, 31, 47),
        (31, 99),
        (0.7, 2.2),
        "channels_last",
        "strided",
    ),
]
DTYPES = [
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_fp64
            or flag_gems.vendor_name in ("mthreads", "iluvatar"),
            reason="Backend or native backward baseline does not support FP64",
        ),
    ),
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_bf16,
            reason="Backend does not support BF16",
        ),
    ),
    pytest.param(
        torch.uint8,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend",
            reason="Native CANN backward does not support uint8",
        ),
    ),
]


class UpsampleNearest2dBackwardBenchmark(base.Benchmark):
    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name != "ascend":
            return super().get_latency(op, *args, **kwargs)

        from torch_npu.npu.utils import get_cann_version

        cann_version = get_cann_version()
        if cann_version != "9.0.0":
            raise RuntimeError(
                "The 35-call profiler contract is validated only for CANN 9.0.0; "
                f"installed version is {cann_version}"
            )
        if base.Config.mode != base.consts.BenchMode.KERNEL:
            return super().get_latency(op, *args, **kwargs)

        # A backward call can launch compute kernels and SDMA copies.
        # Aggregate every device task belonging to each complete call.
        profile_root = Path(
            os.environ.get("FLAGGEMS_ASCEND_PROFILE_DIR", "outputs/ascend_profiles")
        )
        profile_root.mkdir(parents=True, exist_ok=True)
        implementation = "native" if op is self.torch_op else "gems"
        prefix = (
            f"{self.op_name}-{self.case_name}-{args[0].dtype}-" f"{implementation}-"
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
        self.shapes = CASES
        self.shape_desc = (
            "case, input_size, output_size, scales, input_layout, output_layout"
        )

    def get_input_iter(self, dtype):
        for case_name, shape, size, scales, layout, out_layout in self.shapes:
            self.case_name = case_name
            output_shape = (*shape[:2], *size)
            if dtype == torch.uint8:
                grad_output = torch.randint(
                    0, 256, output_shape, dtype=dtype, device=self.device
                )
            else:
                grad_output = torch.randn(output_shape, dtype=dtype, device=self.device)
            if layout == "channels_last":
                grad_output = (
                    grad_output.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
                )
            elif layout == "strided":
                buffer = torch.empty(
                    (*output_shape[:-1], output_shape[-1] * 2),
                    dtype=dtype,
                    device=self.device,
                )
                buffer[..., ::2].copy_(grad_output)
                grad_output = buffer[..., ::2]
            args = (grad_output, size, shape, *scales)
            if self.op_name.endswith(".grad_input"):
                if out_layout == "strided":
                    output = torch.empty(
                        (*shape[:-1], shape[-1] * 2), dtype=dtype, device=self.device
                    )[..., ::2]
                elif out_layout == "channels_last":
                    _, c, h, w = shape
                    output = torch.empty_strided(
                        shape, (c * h * w, 1, c * w, c), dtype=dtype, device=self.device
                    )
                else:
                    output = torch.empty(
                        shape,
                        dtype=dtype,
                        device=self.device,
                    )
                yield (*args, {"grad_input": output})
            else:
                yield args


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(
    _CANN850_ACCURACY_ONLY,
    reason="CANN 8.5.0 is accuracy-only: its profiler does not guarantee the 35-call timing contract",
)
def test_upsample_nearest2d_backward_perf(dtype):
    bench = UpsampleNearest2dBackwardBenchmark(
        op_name="upsample_nearest2d_backward",
        torch_op=torch.ops.aten.upsample_nearest2d_backward.default,
        gems_op=flag_gems.ops.upsample_nearest2d_backward,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.upsample_nearest2d_backward_grad_input
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(
    _CANN850_ACCURACY_ONLY,
    reason="CANN 8.5.0 is accuracy-only: its profiler does not guarantee the 35-call timing contract",
)
def test_upsample_nearest2d_backward_grad_input_perf(dtype):
    bench = UpsampleNearest2dBackwardBenchmark(
        op_name="upsample_nearest2d_backward.grad_input",
        torch_op=torch.ops.aten.upsample_nearest2d_backward.grad_input,
        gems_op=flag_gems.ops.upsample_nearest2d_backward_grad_input,
        dtypes=[dtype],
    )
    bench.run()
