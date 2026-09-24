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

from . import base, consts, utils

MODE_DTYPES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "mthreads"
            and dtype in (torch.float16, torch.bfloat16, torch.int16),
            reason=(
                "MThreads native torch.mode raises 'MUSA error: misaligned address' "
                "during repeated benchmarking for FP16/BF16/INT16 at shapes "
                "(64, 64), (256, 256), and (1024, 1024); skip these dtypes "
                "pending a native backend fix."
            ),
        ),
    )
    for dtype in consts.INT_DTYPES + consts.FLOAT_DTYPES
]


class ModeBenchmark(base.GenericBenchmark2DOnly):
    def get_latency(self, op, *args, **kwargs):
        if (
            flag_gems.vendor_name != "ascend"
            or base.Config.mode != base.consts.BenchMode.KERNEL
        ):
            return super().get_latency(op, *args, **kwargs)
        # A call can contain multiple kernels or an SDMA copy without a kernel.
        # Aggregate every device task, including copies, for the complete call.
        profile_root = Path(
            os.environ.get("FLAGGEMS_ASCEND_PROFILE_DIR", "outputs/ascend_profiles")
        )
        profile_root.mkdir(parents=True, exist_ok=True)
        implementation = "native" if op is self.torch_op else "gems"
        shape = "x".join(str(size) for size in args[0].shape)
        prefix = f"{self.op_name}-{shape}-{args[0].dtype}-{implementation}-"
        profile_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=profile_root))
        # Newer Triton can use MSPTI without exporting CSVs. Request the
        # profiler explicitly when available so the task trace is retained.
        testing = triton.backends.ascend.testing
        profile = getattr(testing, "do_bench_npu_profiler", testing.do_bench_npu)
        profile(
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

    def set_dtypes(self, user_desired_dtypes):
        dtype = self.dtypes[0]
        if user_desired_dtypes and dtype not in user_desired_dtypes:
            pytest.skip(f"{dtype} was not selected by --dtypes")
        self.to_bench_dtypes = [dtype]

    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1},


@pytest.mark.mode
@pytest.mark.skip_native(
    vendors=["ascend"],
    reason="Native torch.mode falls back to CPU on Ascend and has no NPU kernel baseline",
)
@pytest.mark.parametrize("dtype", MODE_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_perf_mode(dtype):
    bench = ModeBenchmark(
        input_fn=_input_fn,
        op_name="mode",
        torch_op=torch.mode,
        gems_op=flag_gems.mode,
        dtypes=[dtype],
    )
    bench.run()
