# Copyright 2026, The FlagOS Contributors.
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

from typing import Generator

import pytest
import torch

from . import base, consts


def _fw_primal_copy_ref(x, level):
    # The native op only runs on inference tensors, so re-create the input
    # inside the InferenceMode block before measuring the copy.
    with torch.inference_mode():
        return torch._fw_primal_copy(
            x.clone(memory_format=torch.contiguous_format), level
        )


def _fw_primal_copy_out_ref(x, level, out):
    with torch.inference_mode():
        return torch.ops.aten._fw_primal_copy.out(
            x.clone(memory_format=torch.contiguous_format),
            level,
            out=out,
        )


class FwPrimalCopyBenchmark(base.UnaryPointwiseBenchmark):
    """Unary copy plus the constant ``level`` tag of ``aten::_fw_primal_copy``."""

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, 0


class FwPrimalCopyOutBenchmark(base.UnaryPointwiseOutBenchmark):
    """``.out`` variant: also passes the constant ``level`` tag."""

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            out = torch.empty_like(inp)
            yield inp, 0, {"out": out}


@pytest.mark.fw_primal_copy
def test_fw_primal_copy():
    bench = FwPrimalCopyBenchmark(
        op_name="fw_primal_copy",
        torch_op=_fw_primal_copy_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fw_primal_copy_out
def test_fw_primal_copy_out():
    bench = FwPrimalCopyOutBenchmark(
        op_name="fw_primal_copy_out",
        torch_op=_fw_primal_copy_out_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
