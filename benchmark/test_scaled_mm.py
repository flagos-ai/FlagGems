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

from functools import lru_cache

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from tests.test_scaled_mm import (
    _device_key,
    _unsupported_reason,
    case_id,
    case_reason,
    cases,
    comparison_tolerance,
    golden,
    make_inputs,
)

from . import base, consts


@lru_cache(None)
def _native_reason(case, shape, use_out, device_key, reference):
    if not hasattr(torch, "_scaled_mm"):
        return "torch._scaled_mm is unavailable"
    args, kwargs = make_inputs(case, shape)
    try:
        if use_out:
            out = torch.empty(
                (shape[0], shape[1]), dtype=case[2] or case[0], device=flag_gems.device
            )
            actual = torch.ops.aten._scaled_mm.out(*args, **kwargs, out=out)
        else:
            actual = torch._scaled_mm(*args, **kwargs)
        torch_device_fn.synchronize()
        tol = comparison_tolerance(actual.dtype)
        torch.testing.assert_close(
            actual.cpu().float(), reference(args, kwargs).float(), rtol=tol, atol=tol
        )
    except AssertionError as exc:
        return f"Native probe disagrees with FP32 golden: {str(exc)[:300]}"
    except Exception as exc:
        return _unsupported_reason(exc)
    return None


def native_reason(case, shape, use_out=False, *, reference):
    return _native_reason(case, shape, use_out, _device_key(), reference)


CORE_SHAPES = [(16, 16, 16), (128, 128, 128), (512, 512, 512)]


def _params(use_out):
    params = []
    for case in cases():
        reason = case_reason(case)
        for shape in CORE_SHAPES:
            marks = []
            if reason:
                marks.append(pytest.mark.skip(reason=f"FP8 capability: {reason}"))
            else:
                baseline_reason = native_reason(case, shape, use_out, reference=golden)
                if baseline_reason:
                    marks.append(
                        pytest.mark.skip_native(
                            vendors=flag_gems.vendor_name, reason=baseline_reason
                        )
                    )
            params.append(
                pytest.param(case, shape, marks=marks, id=f"{case_id(case)}-{shape}")
            )
    return params


class ScaledMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, case, shape, use_out):
        self.case, self.shape, self.use_out = case, shape, use_out
        op_name = "scaled_mm_out" if use_out else "scaled_mm"
        op = torch.ops.aten._scaled_mm.out if use_out else torch._scaled_mm
        super().__init__(op_name, op, dtypes=[case[0]])
        self.set_gems(flag_gems.scaled_mm_out if use_out else flag_gems.scaled_mm)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [self.shape]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype):
        args, kwargs = make_inputs(self.case, self.shape)
        if self.use_out:
            kwargs["out"] = torch.empty(
                (self.shape[0], self.shape[1]),
                dtype=self.case[2] or dtype,
                device=flag_gems.device,
            )
        yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return {
            "fp8_case": case_id(self.case),
            "M_N_K": self.shape,
            "arguments": super().record_shapes(*args, **kwargs),
        }

    def get_tflops(self, op, *args, **kwargs):
        return 2 * args[0].shape[0] * args[0].shape[1] * args[1].shape[1]


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case,shape", _params(False))
def test_scaled_mm_benchmark(case, shape):
    ScaledMMBenchmark(case, shape, False).run()


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("case,shape", _params(True))
def test_scaled_mm_out_benchmark(case, shape):
    ScaledMMBenchmark(case, shape, True).run()
