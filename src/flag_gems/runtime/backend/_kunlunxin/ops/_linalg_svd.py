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

from .linalg_svd import _osj_svd_impl
from .linalg_svdvals import linalg_svdvals

logger = logging.getLogger(__name__)

# Drivers the native CUDA op accepts (``gesvd`` / ``gesvda``; ``gesdd`` is
# rejected). Mirrors the generic ``ops/_linalg_svd.py`` contract.
_VALID_DRIVERS = ("gesvd", "gesvda")


def _linalg_svd(A, full_matrices=False, compute_uv=True, *, driver=None):
    """Triton (XPU) implementation of ``aten::_linalg_svd``.

    The private primitive behind ``torch.linalg.svd`` (``compute_uv=True``) and
    ``torch.linalg.svdvals`` (``compute_uv=False``).  It returns
    ``(U, S, Vh)`` with ``A = U diag(S) Vh``.

    The generic ``ops/_linalg_svd.py`` routes into ``ops/svd.py`` whose kernels
    do not lower on this backend (``ConvertTritonXPUToLLVM`` failures).  We
    instead reuse the already-validated Kunlunxin one-sided Jacobi SVD
    (``linalg_svd``) and singular-value (``linalg_svdvals``) pipelines, which
    are the backend overrides for the public ``torch.linalg.svd`` /
    ``torch.linalg.svdvals`` entry points.
    """
    logger.debug("GEMS_KUNLUNXIN _LINALG_SVD")
    # Validate the driver first, matching native: an unknown driver is rejected
    # regardless of the input dtype.  The Jacobi pipeline has no driver concept
    # -- it always computes the reduced SVD and expands to the full factors,
    # which is what both cuSOLVER drivers do from the caller's perspective -- so
    # a valid driver only has to be accepted, not acted on.
    if driver is not None and driver not in _VALID_DRIVERS:
        raise RuntimeError(
            f"torch.linalg.svd: unknown svd driver {driver} in svd_cusolver "
            "computation. Check doc at "
            "https://pytorch.org/docs/stable/generated/torch.linalg.svd.html"
        )
    if A.dtype != torch.float32:
        raise TypeError(f"_linalg_svd only supports float32 input, got {A.dtype}")

    if not compute_uv:
        # ``aten::_linalg_svd`` leaves U/Vh empty when ``compute_uv`` is False;
        # only the singular values are materialized (descending).
        S = linalg_svdvals(A)
        empty = A.new_empty(0)
        return empty, S, empty

    return _osj_svd_impl(A, full_matrices=full_matrices)
