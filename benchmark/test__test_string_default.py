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
from .generated_operator_utils import OperatorBenchmark

# aten::_test_string_default(Tensor dummy, str a="\"'\\", str b="\"'\\") only
# validates its two optional string defaults and hands the operand back, so a
# workload is one operand plus those strings. The shared shape file has no entry
# for this test-only op, so the shapes below are unioned onto the shared default
# grid instead of replacing it.
_DEFAULT = "\"'\\"

_EXTRA_SHAPES = [
    (256,),
    (2, 19, 7),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# Both argument forms are timed: relying on the schema defaults and passing the
# defaults explicitly.
_VARIANT_KWARGS = {
    "schema_defaults": {},
    "explicit_strings": {"a": _DEFAULT, "b": _DEFAULT},
}

_CANDIDATE_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
]

# Static capability flags, read at import: listing must not create tensors or
# call the operator, and a dtype outside this map is a baseline type.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


_DTYPES = [dtype for dtype in _CANDIDATE_DTYPES if _dtype_supported(dtype)]


def _case_fn(shape, dtype):
    del dtype
    for variant in _VARIANT_KWARGS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"variant": variant},
            builder_args=(shape, variant),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, variant = plan.builder_args
    # The operand is only handed back, never read, so an uninitialized buffer at
    # the case's own shape is enough, and unlike the shared float/int/bool input
    # generator it also covers the float8, int8/uint8/int64 and complex cases.
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, dict(_VARIANT_KWARGS[variant])


class StringDefaultBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Delegate to the shared loader so the core/comprehensive grid (and the
        # extras it merges for COMPREHENSIVE) is kept, then union this op's own
        # shapes on top rather than replacing the shared grid.
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys([tuple(shape) for shape in self.shapes] + _EXTRA_SHAPES)
        )


@pytest.mark.test_string_default
def test_test_string_default():
    bench = StringDefaultBenchmark(
        op_name="_test_string_default",
        torch_op=torch.ops.aten._test_string_default,
        gems_op=getattr(flag_gems, "_test_string_default", None),
        dtypes=_DTYPES,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.run()
