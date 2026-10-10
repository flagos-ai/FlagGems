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

from . import base, consts


def _conj_materialize(x):
    # Materialized reference: force both sides to do O(N) device work.
    # Native torch._conj is a lazy view (conj bit only, zero device work);
    # resolve_conj() materializes it. On the gems side the override already
    # returns a materialized (is_conj()==False) tensor, so resolve_conj() is a
    # cheap no-op there -> both sides measure one O(N) negate-copy pass.
    return torch._conj(x).resolve_conj()


@pytest.mark.conj
def test_conj():
    # _conj only operates on complex dtypes (FLOAT_DTYPES not applicable)
    bench = base.UnaryPointwiseBenchmark(
        op_name="conj",
        torch_op=_conj_materialize,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()
