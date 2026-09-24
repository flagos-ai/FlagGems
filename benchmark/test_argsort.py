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
import shutil
import statistics
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts, utils


def summarize_kernel_rows(rows, warmup, active):
    """Require a fixed kernel sequence, then sum device durations per call."""
    repeats = warmup + active
    if warmup < 0 or active <= 0 or not rows or len(rows) % repeats:
        raise ValueError(
            "NPU profiler rows do not match the requested invocation count"
        )
    kernels = len(rows) // repeats
    sequence = [
        (row["Name"], row["Type"], row.get("Accelerator Core"))
        for row in rows[:kernels]
    ]
    engine_totals = defaultdict(float)
    samples = []
    for invocation in range(repeats):
        group = rows[invocation * kernels : (invocation + 1) * kernels]
        if [
            (row["Name"], row["Type"], row.get("Accelerator Core")) for row in group
        ] != sequence:
            raise ValueError(
                "NPU kernel sequence changed; cannot infer invocation groups"
            )
        durations = [float(row["Duration(us)"]) for row in group]
        if any(not math.isfinite(value) or value < 0 for value in durations):
            raise ValueError("NPU profiler reported an invalid kernel duration")
        if invocation >= warmup:
            samples.append(sum(durations) / 1000)
            for row, duration in zip(group, durations):
                engine_totals[row.get("Accelerator Core") or "unknown"] += duration / (
                    1000 * active
                )
    latency = statistics.mean(samples)
    if latency <= 0:
        raise ValueError("NPU profiler reported a nonpositive operator duration")
    return {
        "protocol": "mean sum of device-kernel durations per invocation",
        "warmup_invocations": warmup,
        "active_invocations": active,
        "kernels_per_invocation": kernels,
        "kernel_sequence": sequence,
        "latency_ms_by_accelerator_core": dict(engine_totals),
        "invocation_ms": samples,
        "latency_ms": latency,
    }


def do_bench_argsort_npu(fn, warmup=5, active=30, context=None):
    """Collect complete invocations independently of Triton timer defaults.

    Native argsort can emit Sort and Cast, while Gems can emit several passes.
    Comparing single kernel-row averages would favor the longer kernel chain.
    Retain the source CSV and grouping report beside other benchmark artifacts.
    """
    import torch_npu

    artifacts = Path("benchmark_profiles") / "argsort" / uuid.uuid4().hex
    artifacts.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="argsort-npu-") as temporary:
        # Compile before recording. Some Triton versions use mspti and ignore
        # profile-directory arguments, so own the CSV-producing collector here.
        fn()
        torch.npu.synchronize()
        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            data_simplification=False,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                temporary, async_mode=False
            ),
            experimental_config=config,
        ):
            for _ in range(warmup + active):
                fn()
                torch.npu.synchronize()
        files = list(Path(temporary).rglob("kernel_details.csv"))
        if len(files) != 1:
            raise RuntimeError("Expected exactly one NPU kernel_details.csv")
        csv_path = artifacts / "kernel_details.csv"
        shutil.copyfile(files[0], csv_path)
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        report = summarize_kernel_rows(rows, warmup, active)
        report["csv"] = str(csv_path.resolve())
        report["context"] = context
        (artifacts / "operator-timing.json").write_text(json.dumps(report, indent=2))
    return report["latency_ms"]


class ArgsortBenchmark(base.GenericBenchmark2DOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_failures = []

    def _measure_input(self, input, case_id=None):
        # Complete timing independently before checking this exact input shape.
        metric = super()._measure_input(input, case_id=case_id)
        args, kwargs = self.unpack_to_args_kwargs(input)
        inp = args[0]
        try:
            actual = self.gems_op(*args, **kwargs).cpu()
            reference = torch.argsort(inp.cpu(), stable=True, **kwargs)
            correct = (
                actual.dtype == torch.int64
                and actual.shape == reference.shape
                and torch.equal(actual, reference)
            )
            metric.accuracy = float(correct)
            if not correct:
                raise AssertionError("stable argsort indices differ from CPU reference")
        except Exception as exc:
            metric.accuracy = 0.0
            failure = f"dtype={inp.dtype}, shape={tuple(inp.shape)}: {exc}"
            metric.error_msg = f"Accuracy check failed: {failure}"
            self.accuracy_failures.append(failure)
        return metric

    def get_latency(self, op, *args, **kwargs):
        if (
            flag_gems.vendor_name == "ascend"
            and base.Config.mode == consts.BenchMode.KERNEL
        ):
            return do_bench_argsort_npu(
                lambda: op(*args, **kwargs),
                context={
                    "role": "native" if op is self.torch_op else "gems",
                    "shape": list(args[0].shape),
                    "dtype": str(args[0].dtype),
                    "kwargs": kwargs,
                },
            )
        return super().get_latency(op, *args, **kwargs)

    def set_more_shapes(self):
        return [(1024, 1), (1024, 512)]


def _input_fn(shape, dtype, device):
    if dtype in (torch.int8, torch.uint8):
        low, high = (-128, 128) if dtype == torch.int8 else (0, 256)
        inp = torch.randint(low, high, shape, dtype=dtype, device="cpu").to(device)
    elif dtype == torch.int64:
        inp = torch.randint(-(2**60), 2**60, shape, dtype=dtype, device="cpu").to(
            device
        )
    else:
        inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1, "descending": False},


@pytest.mark.argsort
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_argsort():
    bench = ArgsortBenchmark(
        input_fn=_input_fn,
        op_name="argsort",
        torch_op=torch.argsort,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES + consts.EXTRA_INT_DTYPES,
    )
    bench.set_gems(flag_gems.argsort)
    bench.run()
    assert not bench.accuracy_failures, "\n".join(bench.accuracy_failures)
