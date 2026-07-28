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
import triton

from flag_gems.runtime import torch_device_fn

from .nonzero import (
    _dense_block_size,
    _is_dense,
    nonzero,
    nonzero_dense_dimmajor_kernel,
)

logger = logging.getLogger(__name__)


def nonzero_numpy(inp):
    """
    Returns a tuple of 1D tensors, one for each dimension of the input,
    containing the indices of the non-zero elements in that dimension.

    This is equivalent to torch.nonzero(...).T or numpy.nonzero() behavior.
    """
    logger.debug("GEMS_KUNLUNXIN NONZERO_NUMPY")

    inp_ndim = inp.ndim
    inp, inp_bool, prefix_sum, num_nonzeros = _is_dense(inp)
    n_elements = inp.numel()

    # DENSE fast path: write coordinates dim-major into [ndim, N] with stride-1
    # contiguous stores, then unbind(0) gives ndim contiguous [N] views for free.
    if inp_ndim >= 1 and num_nonzeros == n_elements and n_elements < 2**31:
        out = torch.empty(inp_ndim, num_nonzeros, dtype=torch.int64, device=inp.device)
        if n_elements > 0:
            shape_t = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)
            block = _dense_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block),)
            with torch_device_fn.device(inp.device):
                nonzero_dense_dimmajor_kernel[grid](
                    out,
                    n_elements,
                    shape_t,
                    inp_ndim,
                    block,
                    isCloseUnrollControl=True,
                    is_use_mask_zero=True,
                )
        return list(out.unbind(dim=0))

    # SPARSE path: reuse the scatter-based nonzero and unbind the [N, ndim] result.
    out = nonzero(inp, as_tuple=False)
    return list(out.unbind(dim=1))
