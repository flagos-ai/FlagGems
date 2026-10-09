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

DTYPES = [torch.float32, torch.complex64]
if flag_gems.vendor_name == "nvidia":
    DTYPES += [torch.float16, torch.float64, torch.complex32, torch.complex128]

CENTER_DTYPES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "nvidia" and dtype == torch.complex32,
            reason="Native CUDA reflection_pad1d does not support complex32",
        ),
    )
    for dtype in DTYPES
]


class STFTBenchmark(base.GenericBenchmark):
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
        prefix = (
            f"{self.op_name}-{shape}-fft{args[1]}-{args[0].dtype}-{implementation}-"
        )
        profile_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=profile_root))
        # Newer Triton can use MSPTI without exporting CSVs. Request the
        # profiler explicitly when available so the task trace is retained.
        testing = triton.backends.ascend.testing
        profile = getattr(testing, "do_bench_npu_profiler", testing.do_bench_npu)
        # Some Ascend launchers collect pointer shapes through size(), which
        # older TensorWrapper versions omit. Supply only that metadata while
        # profiling, and restore the class even when collection fails.
        with pytest.MonkeyPatch.context() as patch:
            wrapper = triton.runtime.jit.TensorWrapper
            if not hasattr(wrapper, "size"):
                patch.setattr(
                    wrapper, "size", lambda value: value.base.size(), raising=False
                )
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

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(2, 4096)]
        self.shape_desc = "batch, signal length"

    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device, center):
    batch, signal_length = shape
    if dtype.is_complex:
        part = torch.float64 if dtype == torch.complex128 else torch.float32
        inp = torch.complex(
            torch.randn(batch, signal_length, device=device, dtype=part),
            torch.randn(batch, signal_length, device=device, dtype=part),
        ).to(dtype)
    else:
        inp = torch.randn(batch, signal_length, device=device, dtype=dtype)

    # Each dtype/overload has one real/complex applicable full-spectrum case,
    # plus windowed, centered and alignment variants.
    for n_fft, hop, win_length, use_window, normalized, align, onesided in (
        (256, 64, 256, False, False, None, True),
        (512, 128, 512, True, False, False, False),
        (1024, 256, 512, True, True, True, True),
    ):
        window = (
            torch.hann_window(
                win_length,
                device=device,
                dtype=(
                    torch.float64
                    if dtype in (torch.float64, torch.complex128)
                    else (
                        torch.float16
                        if dtype in (torch.float16, torch.complex32)
                        else torch.float32
                    )
                ),
            )
            if use_window
            else None
        )
        kwargs = dict(
            hop_length=hop,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=False if dtype.is_complex else onesided,
            return_complex=True,
            align_to_window=None if center else align,
        )
        if center:
            kwargs.update(center=True, pad_mode="reflect")
        yield inp, n_fft, kwargs


def stft_input_fn(shape, dtype, device):
    yield from _input_fn(shape, dtype, device, center=False)


def stft_center_input_fn(shape, dtype, device):
    yield from _input_fn(shape, dtype, device, center=True)


@pytest.mark.stft
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason="Native TorchMUSA STFT copies frames to CPU for FFT; no device baseline",
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_perf_stft(dtype):
    bench = STFTBenchmark(
        op_name="stft",
        torch_op=torch.ops.aten.stft.default,
        gems_op=flag_gems.stft,
        input_fn=stft_input_fn,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.stft
@pytest.mark.stft_center
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason="Native TorchMUSA STFT copies frames to CPU for FFT; no device baseline",
)
@pytest.mark.parametrize("dtype", CENTER_DTYPES)
def test_perf_stft_center(dtype):
    bench = STFTBenchmark(
        op_name="stft_center",
        torch_op=torch.ops.aten.stft.center,
        gems_op=flag_gems.stft_center,
        input_fn=stft_center_input_fn,
        dtypes=[dtype],
    )
    bench.run()
