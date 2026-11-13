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

import logging

import torch

# gcu400: bind the enflame-local svd (device-type gates, GCU kernel fixes).
# Absolute import per the linalg_matmul/expand_copy backend precedent; the
# generic flag_gems.ops.svd copy stays untouched as the semantic baseline.
from flag_gems.runtime.backend._enflame.gcu400.ops.svd import svd

logger = logging.getLogger(__name__)


def linalg_svd(A, full_matrices=True, *, driver=None):
    """Triton implementation of ``aten::linalg_svd``.

    Delegates to the hand-written Triton SVD kernels in ``svd.py``.

    ``torch.linalg.svd`` returns ``(U, S, Vh)`` such that ``A = U diag(S) Vh``.
    The internal ``svd`` helper follows the ``torch.svd`` convention and returns
    ``(U, S, V)`` (with ``A = U diag(S) V^H``), so we transpose-conjugate ``V``
    to obtain ``Vh``. ``full_matrices=True`` corresponds to ``some=False`` (full
    U/V matrices) in the ``torch.svd`` convention.
    """
    logger.debug("GEMS LINALG_SVD")
    # The Triton SVD kernels only implement float32; higher/lower precision
    # matrices (e.g. float16/bfloat16) are not supported by the backend.
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_svd only supports float32 input, got {A.dtype}")
    U, S, V = svd(A, some=not full_matrices, compute_uv=True)
    Vh = V.mH
    return U, S, Vh
