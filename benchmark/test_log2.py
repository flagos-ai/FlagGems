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

# Vendor backends replace ``flag_gems.log2`` with a forward-only kernel, so the
# FlagGems autograd bridge (and therefore the candidate backward) only exists
# for the generic implementation.
_BACKWARD_BRIDGE = "runtime.backend" not in flag_gems.log2.__module__


@pytest.mark.log2
def test_log2():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log2",
        torch_op=torch.log2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.log2_
def test_log2_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log2_",
        torch_op=torch.log2_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.log2_backward
@pytest.mark.skipif(
    not _BACKWARD_BRIDGE,
    reason="this backend's log2 has no FlagGems autograd bridge",
)
def test_log2_backward():
    """Time the FlagGems autograd backward for ``log2``.

    ``gems_op`` is passed explicitly so the candidate column runs the
    FlagGems autograd.Function instead of falling into the native ATen
    formula that the ATen-level ``torch.log2`` dispatch would keep.
    """
    # For 64x64, Python autograd/launch overhead dominates; this case can be
    # slower than the native path even when large workloads benefit.
    bench = base.UnaryPointwiseBenchmark(
        op_name="log2_backward",
        torch_op=torch.log2,
        gems_op=flag_gems.log2,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
