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

"""Profiler timing of an entire callable, in the benchmark's millisecond unit.

An invocation can launch several device kernels. CSV rows are device events,
not invocation boundaries; never use a warmup/repeat index as a row index.
"""

import csv
import json
import math
import shutil
import tempfile
import time
from pathlib import Path


def device_time_ms(csv_paths, repeat_calls):
    """Mean cumulative device execution time per profiled invocation.

    Warmup/calibration must be outside the capture. CPU dispatch and idle time
    are excluded. This is device work, not multi-stream critical-path latency.
    """
    if type(repeat_calls) is not int or repeat_calls < 1:
        raise ValueError("repeat_calls must be positive")
    paths = list(csv_paths)
    if len(paths) != 1:
        raise RuntimeError("expected one complete single-device kernel_details.csv")
    total_us = 0.0
    count = 0
    devices = set()
    with Path(paths[0]).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "Duration(us)" not in (reader.fieldnames or []):
            raise RuntimeError("profiler CSV lacks Duration(us)")
        for row in reader:
            duration = float(row["Duration(us)"])
            if not math.isfinite(duration) or duration < 0:
                raise RuntimeError("profiler emitted an invalid device duration")
            total_us += duration
            count += 1
            if "Device_id" in row:
                devices.add(row["Device_id"])
    if not count or total_us <= 0 or len(devices) > 1:
        raise RuntimeError("profiler has no valid single-device execution time")
    return total_us / repeat_calls / 1000.0


def npu_profiler_time_ms(fn, *, warmup_ms=25, benchmark_ms=100, artifact_dir=None):
    """Profile every device event from active calls; never fall back to wall time."""
    import inspect
    import torch
    import torch_npu

    if warmup_ms < 0 or benchmark_ms <= 0:
        raise ValueError("invalid profiler sampling budget")
    # Calibration determines counts only; it is not the returned measurement.
    fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    estimate_ms = max((time.perf_counter() - started) * 1000.0 / 5, 1e-6)
    warmup_calls = min(5, max(1, int(warmup_ms / estimate_ms))) if warmup_ms else 0
    repeat_calls = min(30, max(1, int(benchmark_ms / estimate_ms)))
    for _ in range(warmup_calls):
        fn()
    torch.npu.synchronize()

    if artifact_dir is not None:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="flaggems-npu-timing-", dir=artifact_dir))
    handler_options = {}
    parameters = inspect.signature(
        torch_npu.profiler.tensorboard_trace_handler
    ).parameters
    if "async_mode" in parameters:
        handler_options["async_mode"] = False
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(directory), **handler_options
    )
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        data_simplification=True,
    )
    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=handler,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental,
        ):
            for _ in range(repeat_calls):
                fn()
                torch.npu.synchronize()
        value = device_time_ms(directory.rglob("kernel_details.csv"), repeat_calls)
    except Exception as exc:
        # Preserve failed capture evidence rather than reporting event/wall time.
        raise RuntimeError(
            f"Ascend callable timing failed; profiler evidence: {directory}"
        ) from exc
    else:
        (directory / "measurement.json").write_text(
            json.dumps(
                {
                    "repeat_calls": repeat_calls,
                    "warmup_calls": warmup_calls,
                    "device_time_ms": value,
                    "scope": "sum of active device kernel durations per invocation",
                },
                indent=2,
            )
        )
        if artifact_dir is None:
            shutil.rmtree(directory)
        return value
