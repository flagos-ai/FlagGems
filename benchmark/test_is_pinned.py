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

# aten::is_pinned(Tensor self, Device? device=None) -> bool is a storage query:
# the measured work is dispatch plus one allocator lookup. No public Benchmark
# family covers a one-tensor predicate returning a bool, so the two-phase
# GenericBenchmark below is used. The shared shape chain is left untouched
# (caller --shape_file, then core_shapes.yaml, then the framework default and
# comprehensive grids), and inputs are payload-free torch.empty allocations
# because the operator reads no element.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


# A metadata query is meaningful for every dtype the backend can allocate.
_BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float8_e4m3fn, torch.float8_e5m2, torch.float64, torch.complex128]
    )
    if _dtype_supported(dtype)
]

_STORAGE_KINDS = [
    ("backend", None),
    ("plain-host", None),
    ("pinned-host", None),
]
if flag_gems.device != "cpu":
    # The same pinned storage queried through the optional device argument.
    _STORAGE_KINDS += [
        ("pinned-host/backend-location", flag_gems.device),
        ("pinned-host/cpu-location", torch.device("cpu")),
    ]


def _case_fn(shape, dtype):
    del dtype
    for kind, device_arg in _STORAGE_KINDS:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"storage": kind},
            builder_args=(shape, kind, device_arg),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, kind, device_arg = plan.builder_args
    if kind == "backend":
        inp = torch.empty(shape, dtype=dtype, device=device)
    elif kind.startswith("pinned-host"):
        inp = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    else:
        inp = torch.empty(shape, dtype=dtype, device="cpu")
    if device_arg is None:
        return (inp,)
    # unpack_to_args_kwargs turns the trailing dict into call kwargs.
    return inp, {"device": device_arg}


@pytest.mark.is_pinned
def test_is_pinned():
    bench = base.GenericBenchmark(
        op_name="is_pinned",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_pinned,
        gems_op=getattr(flag_gems, "is_pinned", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
