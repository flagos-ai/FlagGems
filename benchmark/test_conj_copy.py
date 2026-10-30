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

import pytest
import torch

import flag_gems

from . import base, consts


class ConjCopyBenchmark(base.UnaryPointwiseBenchmark):
    def build_inputs(self, case):
        if flag_gems.vendor_name != "ascend" or not case.dtype.is_complex:
            return super().build_inputs(case)
        expected_device = torch.device(self.device)
        if expected_device.index is None:
            expected_device = torch.device(
                expected_device.type, base.torch_device_fn.current_device()
            )
        shape = case.builder_args[0].builder_args[0]
        inp = torch.randn(shape, dtype=case.dtype, device="cpu").to(self.device)
        assert inp.device == expected_device
        assert inp.dtype == case.dtype and tuple(inp.shape) == tuple(shape)
        return (inp,)


@pytest.mark.underscore_conj_copy
def test_conj_copy():
    # _conj_copy only operates on complex dtypes (consts.FLOAT_DTYPES not applicable)
    bench = ConjCopyBenchmark(
        op_name="_conj_copy",
        torch_op=torch._conj_copy,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.conj_copy_out
def test_conj_copy_out():
    # _conj_copy.out variant writes the conjugated copy into the `out` tensor.
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="conj_copy_out",
        torch_op=torch.ops.aten._conj_copy.out,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()
