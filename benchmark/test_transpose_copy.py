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

TRANSPOSE_COPY_CASES = [
    ((64, 128), 0, 1, False),
    ((256, 512), 0, -1, False),
    ((64, 128, 256), 0, 2, False),
    ((8, 16, 32, 64), 1, -1, False),
    ((256, 512), 0, 0, False),
    ((128, 256, 64), 0, -1, True),
]


class TransposeCopyBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = TRANSPOSE_COPY_CASES

    def get_input_iter(self, dtype):
        for shape, dim0, dim1, non_contiguous in self.shapes:
            if non_contiguous:
                base_shape = (shape[0] * 2, *shape[1:])
                input = torch.empty(base_shape, dtype=dtype, device=self.device)[::2]
            else:
                input = torch.empty(shape, dtype=dtype, device=self.device)
            yield input, dim0, dim1


@pytest.mark.transpose_copy
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "torch_gcu's native aten::transpose_copy faults the device on these "
        "core shapes (Sip exception); only the FlagGems candidate is timed"
    ),
)
def test_transpose_copy():
    dtypes = consts.FLOAT_DTYPES + [
        dtype for dtype in consts.FP8_DTYPES if dtype is not None
    ]
    bench = TransposeCopyBenchmark(
        op_name="transpose_copy",
        torch_op=torch.ops.aten.transpose_copy.int,
        dtypes=dtypes,
    )
    if flag_gems.vendor_name == "enflame":
        # The native op is what this benchmark normally times (as both the
        # baseline and the candidate, since upstream sets no gems_op); on GCU
        # it faults, so measure the FlagGems kernel instead.  Other vendors
        # keep the upstream behaviour.
        bench.set_gems(flag_gems.transpose_copy)
    bench.run()


def _complex_benchmark_params():
    params = []
    for name in ("complex32", "complex64", "complex128"):
        if not hasattr(torch, name):
            continue
        dtype = getattr(torch, name)
        marks = []
        if (
            flag_gems.vendor_name == "enflame"
            and dtype == torch.complex128
            and not flag_gems.runtime.device.support_fp64
        ):
            # gcu400 has no fp64 datapath: the complex128 copy is emulated and
            # its timing loop does not converge.  Other vendors are untouched.
            marks.append(
                pytest.mark.skip(reason="complex128 requires an fp64 datapath")
            )
            params.append(pytest.param(dtype, marks=marks, id=name))
            continue
        try:
            probe = torch.empty((2, 3), dtype=dtype, device=flag_gems.device)
            torch.ops.aten.transpose_copy.int(probe, 0, 1)
        except (NotImplementedError, RuntimeError, TypeError) as error:
            if not any(
                text in str(error).lower()
                for text in ("not support", "unsupported", "not implemented")
            ):
                raise
            marks.append(
                pytest.mark.skip(reason=f"Native {name} benchmark unavailable: {error}")
            )
        params.append(pytest.param(dtype, marks=marks, id=name))
    return params


@pytest.mark.transpose_copy
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "torch_gcu's native aten::transpose_copy faults the device on complex "
        "dtypes; the FlagGems candidate (which handles complex32/64/128) is "
        "still timed"
    ),
)
@pytest.mark.parametrize("dtype", _complex_benchmark_params())
def test_transpose_copy_complex(dtype):
    bench = TransposeCopyBenchmark(
        op_name="transpose_copy",
        torch_op=torch.ops.aten.transpose_copy.int,
        dtypes=[dtype],
    )
    if flag_gems.vendor_name == "enflame":
        # Same reason as test_transpose_copy: the native op faults on GCU, so
        # time the FlagGems kernel.  Other vendors keep the upstream
        # behaviour (no gems_op).
        bench.set_gems(flag_gems.transpose_copy)
    bench.run()
