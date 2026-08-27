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

from . import base


@pytest.mark.special_shifted_chebyshev_polynomial_u
def test_special_shifted_chebyshev_polynomial_u():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_u",
        torch_op=torch.special.shifted_chebyshev_polynomial_u,
        # shifted_chebyshev_polynomial_u_cuda only supports float32
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_shifted_chebyshev_polynomial_u_
def test_special_shifted_chebyshev_polynomial_u_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_u_",
        torch_op=torch.special.shifted_chebyshev_polynomial_u,
        # shifted_chebyshev_polynomial_u_cuda only supports float32
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
