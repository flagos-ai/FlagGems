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

import torch

from ._conj import _conj


def conj_physical(input: torch.Tensor) -> torch.Tensor:
    """Materialize the complex conjugate with the vendor stride-1 kernel."""
    if not input.is_complex():
        return input
    # Match the generic implementation: resolve a pending conjugate bit
    # first, since torch.view_as_real() (used by _conj) rejects unresolved
    # conj views.
    if input.is_conj():
        input = input.resolve_conj()
    return _conj(input)
