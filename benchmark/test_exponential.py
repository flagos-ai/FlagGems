import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts, utils


class ExponentialBenchmark(base.GenericBenchmark):
    def get_latency(self, op, *args, **kwargs):
        """Kernel-mode latency as the sum of one complete call's device tasks.

        The default ascend path (`triton.backends.ascend.testing._collect_prof_result`)
        averages `active` consecutive kernel rows, which assumes one kernel row per
        call. exponential's torch baseline launches several kernels per call
        (a random fill followed by elementwise transforms, plus a copy), so that
        average lands near one call's total device time divided by the number of
        kernels instead of the per-call time. Aggregate every device task of each
        of the 35 profiled calls instead.
        """
        if (
            flag_gems.vendor_name != "ascend"
            or base.Config.mode != base.consts.BenchMode.KERNEL
        ):
            return super().get_latency(op, *args, **kwargs)

        # A call can launch compute kernels and SDMA copies.
        # Aggregate every device task belonging to each complete call.
        profile_root = Path(
            os.environ.get("FLAGGEMS_ASCEND_PROFILE_DIR", "outputs/ascend_profiles")
        )
        profile_root.mkdir(parents=True, exist_ok=True)
        implementation = "native" if op is self.torch_op else "gems"
        prefix = f"{self.op_name}-{args[0].dtype}-{implementation}-"
        profile_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=profile_root))

        from triton.backends.ascend import testing as ascend_testing

        ascend_testing.do_bench_npu(
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

        # Triton appends a numeric suffix to distinguish compiled variants of the
        # same kernel (`exponential_kernel_0`, `exponential_kernel_1`, ...), which
        # changes across calls without changing the device work. Ignore it when
        # checking that every call issued the same task sequence.
        def task_key(row):
            return (re.sub(r"_\d+$", "", row["kernel_name"]), row["kernel_type"])

        sequence = [task_key(row) for row in rows[:width]]
        durations = []
        for start in range(0, len(rows), width):
            group = rows[start : start + width]
            if [task_key(row) for row in group] != sequence:
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
                    "raw_kernel_names": sorted({row["kernel_name"] for row in rows}),
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


@pytest.mark.exponential
def test_exponential_outplace():
    bench = ExponentialBenchmark(
        op_name="exponential",
        input_fn=utils.unary_input_fn,
        torch_op=torch.ops.aten.exponential,
        gems_op=flag_gems.exponential,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
