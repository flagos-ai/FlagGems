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

from flag_gems.ops.erfinv_backward import ErfinvFunction

from . import base, consts


class ErfinvBenchmark(base.UnaryPointwiseBenchmark):
    """``erfinv`` inputs must stay inside (-1, 1).

    The default ``UnaryPointwiseBenchmark`` inputs come from ``torch.randn``,
    which leaves the domain about a third of the time; there the gradient is
    ``nan`` and the comparison would time a degenerate NaN path. Sampling
    uniformly inside the domain keeps the timed workload representative.
    """

    @staticmethod
    def _make_input(shape, dtype, device):
        return (
            torch.empty(shape, dtype=dtype, device=device)
            .uniform_(-0.99, 0.99)
            .to(dtype)
        )

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield self._make_input(shape, cur_dtype, self.device),

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return (self._make_input(shape, case.dtype, self.device),)


@pytest.mark.erfinv_backward
def test_erfinv_backward():
    # ``aten::erfinv`` keeps its own autograd formula node (erfinv/pow/exp/
    # mul), so drive the gradient through the FlagGems autograd bridge
    # explicitly -- otherwise this would time the native five-kernel chain,
    # not this kernel.
    bench = ErfinvBenchmark(
        op_name="erfinv_backward",
        torch_op=torch.erfinv,
        gems_op=ErfinvFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
