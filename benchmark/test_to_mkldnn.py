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

# aten::to_mkldnn converts a dense CPU tensor into an opaque oneDNN (mkldnn)
# tensor and has no accelerator kernel, so the workload stays on CPU and the
# candidate is measured against the same native call. Allocation happens in the
# builder, outside the timed region, and its contents are never read.
TO_MKLDNN_EXTRA_SHAPES = [
    (256,),
    (2, 19, 7),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
    (128, 64, 56, 56),
]

# Every legal optional-dtype branch per input dtype; None keeps the input dtype.
# int8/uint8 input may not change dtype: "For int8, uint8 cpu_tensor input, we
# should not change the dtype".
DTYPE_BRANCHES = {
    torch.float32: (
        None,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int8,
        torch.uint8,
    ),
    torch.bfloat16: (
        None,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int8,
        torch.uint8,
    ),
    torch.float16: (
        None,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int8,
        torch.uint8,
    ),
    torch.int8: (None, torch.int8),
    torch.uint8: (None, torch.uint8),
}

TO_MKLDNN_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int8,
    torch.uint8,
]


def _case_fn(shape, dtype):
    for out_dtype in DTYPE_BRANCHES[dtype]:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"dtype": None if out_dtype is None else str(out_dtype)},
            builder_args=(shape, out_dtype),
        )


def _build_inputs_fn(plan, dtype, device):
    del device  # CPU-only operator: the workload is allocated on CPU
    shape, out_dtype = plan.builder_args
    inp = torch.empty(shape, dtype=dtype)
    # Flat positional arguments plus a trailing kwargs dict; a None dtype is
    # passed positionally, an explicit one as op(inp, dtype).
    return inp, out_dtype, {}


class ToMkldnnBenchmark(OperatorBenchmark):
    """Keeps the shared shape sets and adds reorder-heavy CPU shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # A 0-dim input has no reorder primitive ("could not create a primitive
        # descriptor for the reorder primitive"), so it is excluded here.
        shapes = [tuple(shape) for shape in self.shapes if shape]
        self.shapes = list(dict.fromkeys(shapes + TO_MKLDNN_EXTRA_SHAPES))


@pytest.mark.to_mkldnn
def test_to_mkldnn():
    bench = ToMkldnnBenchmark(
        op_name="to_mkldnn",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_mkldnn,
        gems_op=getattr(flag_gems, "to_mkldnn", None),
        dtypes=TO_MKLDNN_DTYPES,
    )
    bench.run()
