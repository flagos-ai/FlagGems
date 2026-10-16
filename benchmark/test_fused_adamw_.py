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


import pytest
import torch

import flag_gems

from . import base
from .consts import BenchmarkCasePlan

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
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(range(len(ADAMW_WORKLOADS)))
        self.shape_desc = "parameter tensor list, AMSGrad"

    def set_dtypes(self, user_desired_dtypes):
        # Parametrized tests must honor --dtypes without asking each single-
        # dtype benchmark to accept the other selected dtypes.
        if user_desired_dtypes and self.dtypes[0] not in user_desired_dtypes:
            pytest.skip(f"{self.dtypes[0]} was not selected by --dtypes")
        self.to_bench_dtypes = self.dtypes


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
