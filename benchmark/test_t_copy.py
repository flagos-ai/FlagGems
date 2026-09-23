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

from typing import Generator

import pytest
import torch

from . import base, consts, utils


class TCopyBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) == 2:
                inp = utils.generate_tensor_input(shape, dtype, self.device)
                yield inp,


@pytest.mark.t_copy
def test_t_copy():
    bench = TCopyBenchmark(
        op_name="t_copy",
        torch_op=torch.ops.aten.t_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class TCopyOutBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) == 2:
                inp = utils.generate_tensor_input(shape, dtype, self.device)
                out_shape = (shape[1], shape[0])
                out = torch.empty(out_shape, dtype=dtype, device=self.device)
                yield inp, {"out": out}


@pytest.mark.t_copy_out
def test_t_copy_out():
    bench = TCopyOutBenchmark(
        op_name="t_copy_out",
        torch_op=torch.ops.aten.t_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
