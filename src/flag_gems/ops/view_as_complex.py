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


def view_as_complex(A: torch.Tensor) -> torch.Tensor:
    r"""Return a view of the real input tensor as a complex tensor.

    The input tensor must have its last dimension of size 2, representing
    the real and imaginary parts of complex numbers.

    Args:
        A (torch.Tensor): Input float tensor with last dimension of size 2.

    Returns:
        torch.Tensor: A complex tensor view.
    """
    logger.debug("GEMS VIEW_AS_COMPLEX")

    # Validate input - support float16, float32, and float64
    # float16 -> complex32, float32 -> complex64, float64 -> complex128
    assert A.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
    ), f"view_as_complex only supports float16/float32/float64, got {A.dtype}"

    # Validate last dimension
    if A.size(-1) != 2:
        raise RuntimeError(
            f"view_as_complex expects a tensor with last dimension of size 2, "
            f"but got size {A.size(-1)}"
        )

    # Map real dtype to complex dtype
    if A.dtype == torch.float16:
        complex_dtype = torch.complex32
    elif A.dtype == torch.float32:
        complex_dtype = torch.complex64
    else:  # float64
        complex_dtype = torch.complex128

    # Reinterpret memory as complex dtype via Tensor.view(dtype).
    # This calls aten.view.dtype (not aten.view_as_complex), avoiding
    # FlagGems dispatch interception and infinite recursion in use_gems().
    # view(complex_dtype) merges the last dim-2 into one complex element,
    # producing shape [..., 1]; squeeze(-1) removes that size-1 trailing dim
    # to match the output shape of torch.view_as_complex.
    return A.view(complex_dtype).squeeze(-1)
