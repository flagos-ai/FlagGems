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

from flag_gems.runtime import torch_device_fn

from .copy import copy_
from .linalg_lu_factor import _check_linalg_lu_factor, _linalg_lu_factor
from .lu_unpack import lu_unpack

logger = logging.getLogger(__name__)


def _resolve_linalg_lu_out_args(P, L, U, out):
    if out is not None:
        if P is not None or L is not None or U is not None:
            raise TypeError("linalg_lu(): out and P/L/U cannot both be set")
        if len(out) != 3:
            raise TypeError(
                "linalg_lu(): out must be a tuple of 3 tensors, " f"got {len(out)}"
            )
        return out
    if P is None or L is None or U is None:
        raise TypeError("linalg_lu(): P, L and U must all be provided for out variant")
    return P, L, U


def linalg_lu(input, *, pivot=True):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU")
    _check_linalg_lu_factor(input, pivot)
    with torch_device_fn.device(input.device):
        lu, pivots = _linalg_lu_factor(input, pivot)
        P, L, U = lu_unpack(lu, pivots, unpack_data=True, unpack_pivots=True)
    return P, L, U


def linalg_lu_out(input, *, pivot=True, P=None, L=None, U=None, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU.OUT")
    _check_linalg_lu_factor(input, pivot)
    p_out, l_out, u_out = _resolve_linalg_lu_out_args(P, L, U, out)
    with torch_device_fn.device(input.device):
        lu, pivots = _linalg_lu_factor(input, pivot)
        P_res, L_res, U_res = lu_unpack(
            lu, pivots, unpack_data=True, unpack_pivots=True
        )

    if P_res.numel() > 0:
        if p_out.numel() != P_res.numel():
            p_out.resize_(P_res.shape)
        copy_(p_out, P_res)
    else:
        p_out.resize_((0,))
    if l_out.shape != L_res.shape:
        l_out.resize_(L_res.shape)
    copy_(l_out, L_res)
    if u_out.shape != U_res.shape:
        u_out.resize_(U_res.shape)
    copy_(u_out, U_res)
    return (p_out, l_out, u_out)
