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

logger = logging.getLogger(__name__)

_COMPLEX_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def view_as_complex(A: torch.Tensor) -> torch.Tensor:
    r"""Return a view of the real input tensor as a complex tensor.

    The input tensor must have its last dimension of size 2, representing
    the real and imaginary parts of complex numbers.

    Args:
        A (torch.Tensor): Input float tensor with last dimension of size 2.

    Returns:
        torch.Tensor: A complex tensor view.
    """
    logger.debug("GEMS_ENFLAME VIEW_AS_COMPLEX")

    # Map real dtype to complex dtype (float16 -> complex32,
    # float32 -> complex64, float64 -> complex128); a single dict lookup
    # replaces the tuple-membership assert plus the if/elif chain.
    complex_dtype = _COMPLEX_DTYPE.get(A.dtype)
    if complex_dtype is None:
        raise RuntimeError(
            f"view_as_complex only supports float16/float32/float64, got {A.dtype}"
        )

    # Validate last dimension
    if A.shape[-1] != 2:
        raise RuntimeError(
            f"view_as_complex expects a tensor with last dimension of size 2, "
            f"but got size {A.shape[-1]}"
        )

    # Reinterpret memory as complex dtype via Tensor.view(dtype).
    # This calls aten.view.dtype (not aten.view_as_complex), avoiding
    # FlagGems dispatch interception and infinite recursion in use_gems().
    # view(complex_dtype) merges the last dim-2 into one complex element,
    # producing shape [..., 1]; dropping that trailing size-1 dim via
    # select(-1, 0) matches the output shape of torch.view_as_complex.
    #
    # select is used instead of squeeze deliberately: FlagGems registers
    # aten::squeeze / aten::squeeze.dim, so a squeeze here would chain a
    # second python-kernel dispatch hop (~2.5us on the gcu400 host path),
    # while aten::select stays native under use_gems(). select is a pure
    # view op for a size-1 trailing dim, so the result is identical to
    # squeeze(-1) for every rank, stride and dtype view(dtype) accepts.
    return A.view(complex_dtype).select(-1, 0)
