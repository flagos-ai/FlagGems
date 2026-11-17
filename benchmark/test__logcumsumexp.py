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

from . import base, consts


def _logcumsumexp_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp, 1


def _logcumsumexp_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={"dim": 1}, builder_args=(dims,)
    )


def _logcumsumexp_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return next(_logcumsumexp_input_fn(shape, dtype, device))


def _musa_bf16_without_native(inp):
    return (
        flag_gems.vendor_name == "mthreads"
        and inp.dtype == torch.bfloat16
        and inp.numel() != 0
    )


class LogcumsumexpBenchmark(base.GenericBenchmark2DOnly):
    def build_inputs(self, case):
        inputs = super().build_inputs(case)
        args, kwargs = self.unpack_to_args_kwargs(inputs)
        # Candidate-only traversal does not consume a native timing record.
        if _musa_bf16_without_native(args[0]) and not (
            base.Config.preflight_only or base.Config.profile_only
        ):
            if base.Config.reference_only:
                pytest.skip("MUSA native logcumsumexp lacks nonempty BFloat16")
            context = self._capture_external_baseline(args, kwargs)
            if context is not None:
                baseline, call, named_call = context
                record, _, _ = baseline.lookup(
                    self.op_name, case.dtype, call, named_call, target_device=self.device
                )
                if record is not None:
                    return inputs
            pytest.skip("MUSA native logcumsumexp lacks BFloat16 and no matching baseline is loaded")
        return inputs

    def _measure_input(self, inputs, case_id=None):
        args, _ = self.unpack_to_args_kwargs(inputs)
        if not _musa_bf16_without_native(args[0]):
            return super()._measure_input(inputs, case_id=case_id)
        previous = base.Config.skip_native
        base.Config.skip_native = True
        try:
            return super()._measure_input(inputs, case_id=case_id)
        finally:
            base.Config.skip_native = previous


@pytest.mark.underscore_logcumsumexp
def test__logcumsumexp():
    bench = LogcumsumexpBenchmark(
        op_name="_logcumsumexp",
        case_fn=_logcumsumexp_case_fn,
        build_inputs_fn=_logcumsumexp_build_inputs_fn,
        torch_op=torch._logcumsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _logcumsumexp_out_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    out = torch.empty_like(inp)
    yield inp, 1, {"out": out}


@pytest.mark.underscore_logcumsumexp_out
def test__logcumsumexp_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="_logcumsumexp_out",
        input_fn=_logcumsumexp_out_input_fn,
        torch_op=torch.ops.aten._logcumsumexp.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
