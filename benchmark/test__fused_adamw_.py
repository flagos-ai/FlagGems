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

import json
import logging
import math
import statistics

import pytest
import torch

import flag_gems

from . import base
from .conftest import TEST_RESULTS, Config, emit_record_logger
from .consts import BenchmarkCasePlan, BenchMode

LOGGER = logging.getLogger(__name__)

# Both native overloads unconditionally reject the call in this vendor source:
# https://github.com/MooreThreads/torch_musa/blob/
# 5e92b0d87ed2981f470391a1bed97a4d0f06aa32/torch_musa/csrc/aten/ops/FusedAdam.cpp#L45-L84
MTHREADS_NATIVE_UNAVAILABLE = pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason=(
        "native ATen _fused_adamw_ overloads are unsupported by "
        "torch_musa FusedAdamWKernel (source 5e92b0d87ed2)"
    ),
)

# Pin all workloads before timing, including launch-bound and bandwidth-bound
# lists. The largest AMSGrad case allocates six 2**24-element buffers (768 MiB
# for float64). Both levels and both overloads use exactly the same case set.
ADAMW_WORKLOADS = (
    ("single_tail", ((257,),)),
    ("single_medium", ((65536,),)),
    ("single_huge", ((2**24,),)),
    ("many_small_64", ((257,),) * 64),
    ("many_small_256", ((33,),) * 256),
    (
        "heterogeneous",
        (
            (512, 512),
            (2048, 512),
            (2048,),
            (512, 2048),
            (512,),
            (4096, 512),
            (512, 4096),
            (512,),
            (1000,),
            (7, 7, 7),
        ),
    ),
)
AMSGRAD_VALUES = (False, True)
ADAMW_DTYPES = (
    pytest.param(torch.float16, id="f16"),
    pytest.param(
        torch.bfloat16,
        id="bf16",
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_bf16,
            reason="the backend does not support bfloat16",
        ),
    ),
    pytest.param(torch.float32, id="f32"),
    pytest.param(
        torch.float64,
        id="f64",
        marks=pytest.mark.skipif(
            flag_gems.vendor_name in ("ascend", "iluvatar", "mthreads")
            or not flag_gems.runtime.device.support_fp64,
            reason="Backend fused AdamW policy excludes float64",
        ),
    ),
)


class FusedAdamWBenchmark(base.GenericBenchmark):
    def _time_callable(self, fn, xs):
        if flag_gems.vendor_name != "ascend" or Config.mode != BenchMode.KERNEL:
            return super()._time_callable(fn, xs)
        # Ascend do_bench_npu averages CSV kernel rows, which does not measure
        # a multi-launch optimizer call. Device events enclose every launch of
        # the complete native/candidate operation on the same stream. Timings
        # include stream gaps between launches; no per-kernel mean is used.
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        end.synchronize()
        estimate_ms = start.elapsed_time(end) / 5
        if not math.isfinite(estimate_ms) or estimate_ms <= 0:
            pytest.fail("AdamW NPU event timing must be finite and positive")
        for _ in range(max(1, int(Config.warm_up / estimate_ms))):
            fn()
        torch.npu.synchronize()
        repetitions = max(1, int(Config.repetition / (20 * estimate_ms)))
        samples = []
        for _ in range(20):
            start.record()
            for _ in range(repetitions):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / repetitions)
        return statistics.median(samples)

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(range(len(ADAMW_WORKLOADS)))
        self.shape_desc = "parameter tensor list, AMSGrad"

    def set_dtypes(self, user_desired_dtypes):
        # Parametrized tests must honor --dtypes without asking each single-
        # dtype benchmark to accept the other selected dtypes.
        if user_desired_dtypes and self.dtypes[0] not in user_desired_dtypes:
            pytest.skip(f"{self.dtypes[0]} was not selected by --dtypes")
        self.to_bench_dtypes = self.dtypes

    def _emit_result(self, dtype, metrics):
        result = super()._emit_result(dtype, metrics)
        expected_ids = {case.case_id for case in self.get_case_iter(dtype)}
        observed_ids = {metric.case_id for metric in metrics}
        summary = {
            "backend": flag_gems.vendor_name,
            "overload": self.op_name,
            "dtype": str(dtype),
            "mode": Config.mode.value,
            "timing_method": (
                "npu_event_whole_operation"
                if flag_gems.vendor_name == "ascend" and Config.mode == BenchMode.KERNEL
                else Config.mode.value
            ),
            "level": Config.bench_level.value,
            "expected_cases": len(expected_ids),
            "measured_cases": len(metrics),
            "geomean_speedup": None,
        }
        if observed_ids != expected_ids or len(metrics) != len(expected_ids):
            summary["status"] = "INCOMPLETE"
        elif any(
            metric.latency_base is None or metric.latency is None for metric in metrics
        ):
            summary["status"] = "NO_NATIVE_COMPARISON"
        else:
            ratios = [metric.latency_base / metric.latency for metric in metrics]
            if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
                pytest.fail("AdamW timings must give finite positive speedups")
            summary["status"] = "COMPLETE"
            summary["geomean_speedup"] = math.exp(
                math.fsum(math.log(ratio) for ratio in ratios) / len(ratios)
            )
        LOGGER.info("AdamW performance summary: %s", json.dumps(summary))
        emit_record_logger(json.dumps(summary))
        if Config.record_json:
            TEST_RESULTS[self.op_name].setdefault("geomean", {})[str(dtype)] = summary
        return result


def _case_fn(workload_index, dtype):
    name, shapes = ADAMW_WORKLOADS[workload_index]
    for amsgrad in AMSGRAD_VALUES:
        yield BenchmarkCasePlan(
            shape={"params": shapes},
            params={"workload": name, "amsgrad": amsgrad},
            builder_args=(shapes, amsgrad),
        )


def _build_inputs(plan, dtype, device, *, tensor_lr):
    shapes, amsgrad = plan.builder_args
    # Native and candidate are timed repeatedly on the same inputs. Zero
    # parameters/gradients/moments remain zero with positive eps, even under
    # nonzero weight decay; state_steps are read-only and remain at one. This
    # avoids reset kernels in the timing and preserves identical initial state.
    params = [torch.zeros(shape, dtype=dtype, device=device) for shape in shapes]
    grads = [torch.zeros_like(param) for param in params]
    exp_avgs = [torch.zeros_like(param) for param in params]
    exp_avg_sqs = [torch.zeros_like(param) for param in params]
    max_exp_avg_sqs = [torch.zeros_like(param) for param in params] if amsgrad else []
    state_steps = [torch.ones((), dtype=torch.float32, device=device) for _ in params]
    kwargs = {
        "lr": (
            torch.tensor(0.001, dtype=torch.float32, device=device)
            if tensor_lr
            else 0.001
        ),
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "amsgrad": amsgrad,
        "maximize": False,
    }
    return params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, kwargs


@pytest.mark.fused_adamw_
@pytest.mark.parametrize("dtype", ADAMW_DTYPES)
@MTHREADS_NATIVE_UNAVAILABLE
def test_fused_adamw_(dtype):
    bench = FusedAdamWBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=lambda plan, dtype, device: _build_inputs(
            plan, dtype, device, tensor_lr=False
        ),
        op_name="fused_adamw_",
        torch_op=torch.ops.aten._fused_adamw_.default,
        gems_op=flag_gems._fused_adamw_,
        dtypes=[dtype],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("dtype", ADAMW_DTYPES)
@MTHREADS_NATIVE_UNAVAILABLE
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    # 2026-10-08 CANN 9.0/8.5 with torch-npu 2.10/2.9 emit npu_cpu_fallback
    # warnings for this exact overload. The CANN 9.0 native profiler also
    # records seven _to_cpu calls. Scalar-LR has its own NPU kernel.
    reason="torch-npu native aten::_fused_adamw_.tensor_lr falls back to CPU",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "iluvatar",
    # Native device-LR probe on BI-V150, 2026-10-08: Torch 2.7.1+corex.4.4.0,
    # driver 4.4.0, f16/bf16/f32. Raw evidence: native-probe-v3a-qualified/
    # native-support.json in the adamw-20261008-iluvatar runtime task logs.
    reason=(
        "CoreX 4.4 native aten::_fused_adamw_.tensor_lr raises "
        "RuntimeError: CUDA error: invalid device function"
    ),
)
def test_fused_adamw__tensor_lr(dtype):
    bench = FusedAdamWBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=lambda plan, dtype, device: _build_inputs(
            plan, dtype, device, tensor_lr=True
        ),
        op_name="fused_adamw__tensor_lr",
        torch_op=torch.ops.aten._fused_adamw_.tensor_lr,
        gems_op=flag_gems._fused_adamw__tensor_lr,
        dtypes=[dtype],
        is_inplace=True,
    )
    bench.run()
