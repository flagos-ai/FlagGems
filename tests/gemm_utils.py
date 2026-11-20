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

if __package__ == "flaggems_tests":
    from flaggems_benchmark.gemm_shapes import accuracy_cases, accuracy_shapes
else:
    from benchmark.gemm_shapes import accuracy_cases, accuracy_shapes

_BF16_SKIP = pytest.mark.skipif(
    not flag_gems.runtime.device.support_bf16,
    reason="this backend does not support bfloat16",
)
GEMM_DTYPES = [torch.float16, pytest.param(torch.bfloat16, marks=_BF16_SKIP)]

_MM_FP32_CASES = ((32, 64, 128, False), (33, 65, 129, True))
_BMM_FP32_CASES = ((2, 32, 64, 128, False), (4, 33, 65, 129, True))
_FP32_CASES = {
    "mm": _MM_FP32_CASES,
    "addmm": _MM_FP32_CASES,
    "mv": ((64, 128, False), (65, 129, True)),
    "bmm": _BMM_FP32_CASES,
    "baddbmm": _BMM_FP32_CASES,
}
_TSINGMICRO_FP32_ISSUES = {"mm": 2834, "addmm": 3794, "bmm": 2799, "baddbmm": 3794}


def accuracy_dtype_cases(op, quick, *, scalars=None):
    """Combine the catalog's low-precision matrix with targeted FP32 accuracy cases."""
    scalar_args = [()] if scalars is None else [(scalar,) for scalar in scalars]
    params = []
    for case in accuracy_cases(op, quick):
        for args in scalar_args:
            params.append(pytest.param(torch.float16, *args, *case))
            params.append(pytest.param(torch.bfloat16, *args, *case, marks=_BF16_SKIP))

    fp32_cases = _FP32_CASES[op]
    if flag_gems.vendor_name == "thead":
        # Use small FP32 cases with verified cold-cache compilation on PPU.
        if op == "mm":
            fp32_cases = ((1, 1, 32, False),)
        elif op == "addmm":
            fp32_cases = ((3, 5, 32, False),)
        elif op == "baddbmm":
            fp32_cases = fp32_cases[:1]
    issue = _TSINGMICRO_FP32_ISSUES.get(op)
    marks = (
        (pytest.mark.skip(reason=f"Issue #{issue}: FP32 {op} on Tsingmicro"),)
        if flag_gems.vendor_name == "tsingmicro" and issue is not None
        else ()
    )
    for case in fp32_cases:
        transposed = case[-1]
        args = () if scalars is None else ((-0.5 if transposed else 0.75),)
        layout = "transposed" if transposed else "contiguous"
        params.append(
            pytest.param(torch.float32, *args, *case, id=f"fp32-{layout}", marks=marks)
        )
    return params


__all__ = ["GEMM_DTYPES", "accuracy_cases", "accuracy_dtype_cases", "accuracy_shapes"]
