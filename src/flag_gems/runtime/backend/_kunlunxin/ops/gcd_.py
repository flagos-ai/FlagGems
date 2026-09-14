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

# Kunlunxin (XPU) override for the in-place gcd_ variant. Reuses the
# XPU-safe binary GCD kernels from the gcd override (the generic
# `flag_gems/ops/gcd.py` kernels cannot compile on XPU -- see the notes in
# `_kunlunxin/ops/gcd.py`).

import logging

import torch

from flag_gems.ops.gcd import _materialize_inputs

from .gcd import _launch_gcd

logger = logging.getLogger(__name__)


def gcd_(A: torch.Tensor, B: torch.Tensor):
    """In-place GCD: A = gcd(A, B), reusing the XPU binary GCD kernel."""
    logger.debug("GEMS_KUNLUNXIN GCD_")
    if A.numel() == 0:
        # Short-circuit empty in-place input: the generic _materialize_inputs
        # path goes through the gems broadcast_tensors override, which fails
        # on 0-size tensors. ATen leaves the empty A unchanged here.
        return A
    lhs, rhs, promoted_dtype = _materialize_inputs(A, B)
    # Compute GCD into a flat buffer, then write back in-place.
    flat_out = torch.empty(lhs.numel(), dtype=promoted_dtype, device=A.device)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), flat_out)
    # NOTE (kunlunxin/XPU): write back through the ATen `_copy_from`
    # primitive -- gems overrides `copy_`/`copy` but never `_copy_from`, so
    # this reaches the vendor's native strided-copy engine and avoids
    # nested gems copy_ dispatch.
    torch.ops.aten._copy_from(flat_out.view(A.shape), A, False)
    return A
