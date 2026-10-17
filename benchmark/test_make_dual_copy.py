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


def _make_dual_copy_ref(primal, tangent, level):
    # The native op only runs on inference tensors, so re-create both inputs
    # inside the InferenceMode block before measuring the dual construction.
    with torch.inference_mode():
        p = primal.clone(memory_format=torch.contiguous_format)
        t = tangent.clone(memory_format=torch.contiguous_format)
        return torch._make_dual_copy(p, t, level)


def _make_dual_copy_out_ref(primal, tangent, level, out):
    with torch.inference_mode():
        p = primal.clone(memory_format=torch.contiguous_format)
        t = tangent.clone(memory_format=torch.contiguous_format)
        return torch.ops.aten._make_dual_copy.out(p, t, level, out=out.clone())


class MakeDualCopyBenchmark(base.Benchmark):
    """``aten::_make_dual_copy`` over a (primal, tangent) tensor pair.

    The workload is the primal copy plus the tangent attachment, so both
    operands share the same shapes axis.
    """

    def set_shapes(self, shape_file_path=None):
        # Unary pointwise core shapes: the dual value holds two same-size
        # operands, so one shape axis describes both inputs.
        self.shapes = [
            (1024 * 1024 * 1024,),
            (64, 64),
            (4096, 4096),
            (64, 512, 512),
            (1024, 1024, 1024),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            primal = base.generate_tensor_input(shape, cur_dtype, self.device)
            tangent = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield primal, tangent, 0


class MakeDualCopyOutBenchmark(MakeDualCopyBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            primal = base.generate_tensor_input(shape, cur_dtype, self.device)
            tangent = base.generate_tensor_input(shape, cur_dtype, self.device)
            out = torch.empty_like(primal)
            yield primal, tangent, 0, {"out": out}


@pytest.mark.make_dual_copy
def test_make_dual_copy():
    bench = MakeDualCopyBenchmark(
        op_name="make_dual_copy",
        torch_op=_make_dual_copy_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.make_dual_copy_out
def test_make_dual_copy_out():
    bench = MakeDualCopyOutBenchmark(
        op_name="make_dual_copy_out",
        torch_op=_make_dual_copy_out_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
