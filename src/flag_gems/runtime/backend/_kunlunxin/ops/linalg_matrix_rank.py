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
import struct
import warnings

import torch

from flag_gems.runtime import device as runtime_device

from .linalg_svdvals import linalg_svdvals
from .mm import mm as _mm

logger = logging.getLogger(__name__)

_SVDVALS_MAX_DIM = 512
_SIGN_ITERS = 80
_POWER_ITERS = 60


def _native_fp64_supported():
    return getattr(runtime_device, "support_fp64", True)


def _expand_tolerance(value, batch_shape, input, name):
    tol_dtype = torch.float64 if _native_fp64_supported() else torch.float32
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} tensor of complex type is not "
                f"supported. Got {value.dtype}"
            )
        if value.device != input.device:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: Expected {name} and input tensors to "
                f"be on the same device, but got {name} on {value.device} and "
                f"input on {input.device}"
            )
        try:
            value = value.expand(batch_shape)
        except RuntimeError as error:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} with shape {tuple(value.shape)} "
                f"is not broadcastable to batch shape {tuple(batch_shape)}"
            ) from error
        return value.to(dtype=tol_dtype).contiguous()

    raise TypeError(f"torch.linalg.matrix_rank: {name} must be a float or Tensor")


def _scalar_tolerance(value, name):
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"torch.linalg.matrix_rank: {name} must be a float or Tensor"
        ) from error


def _is_exact_float32(value):
    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        return False
    return rounded == value


def _prepare_tolerances(input, atol, rtol):
    batch_shape = input.shape[:-2]
    atol_is_set = atol is not None
    if atol is None:
        atol_val = 0.0
    elif isinstance(atol, torch.Tensor):
        atol_val = _expand_tolerance(atol, batch_shape, input, "atol")
    else:
        atol_val = _scalar_tolerance(atol, "atol")

    if isinstance(rtol, torch.Tensor):
        rtol_val = _expand_tolerance(rtol, batch_shape, input, "rtol")
    elif rtol is not None:
        rtol_val = _scalar_tolerance(rtol, "rtol")
    else:
        default_rtol = max(input.shape[-2:]) * torch.finfo(input.dtype).eps
        if not atol_is_set:
            rtol_val = default_rtol
        elif isinstance(atol_val, torch.Tensor):
            rtol_val = torch.where(
                atol_val > 0,
                torch.zeros_like(atol_val),
                torch.full_like(atol_val, default_rtol),
            )
        else:
            rtol_val = 0.0 if atol_val > 0 else default_rtol
    if isinstance(rtol_val, torch.Tensor):
        rtol_val = rtol_val.contiguous()
    return atol_val, rtol_val


def _check_input(input, hermitian):
    if input.ndim < 2:
        raise RuntimeError(
            "torch.linalg.matrix_rank: input must have at least 2 dimensions"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank currently supports float32 and float64 "
            f"real inputs only; got {input.dtype}"
        )
    if input.dtype == torch.float64 and not _native_fp64_supported():
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank: float64 input requires native FP64 "
            "support, which this device does not provide"
        )
    if hermitian and input.shape[-2] != input.shape[-1]:
        raise RuntimeError(
            "torch.linalg.matrix_rank: A must be batches of square matrices when "
            "hermitian=True"
        )


def _empty_matrix_rank(input, output_shape):
    return torch.zeros(output_shape, dtype=torch.int64, device=input.device)


def _power_iter_specnorm(M, iters=_POWER_ITERS):
    d = M.shape[0]
    v = torch.randn(d, 1, device=M.device, dtype=torch.float32)
    nv = v.norm()
    if nv == 0.0:
        return 0.0
    v = v / nv
    for _ in range(iters):
        w = _mm(M, v)
        nv = w.norm()
        if nv == 0.0:
            return 0.0
        v = w / nv
    return _mm(M, v).norm().item()


def _count_pos_sign(M, iters=_SIGN_ITERS):
    d = M.shape[0]
    rho = _power_iter_specnorm(M)
    if rho == 0.0:
        return 0.0
    rho *= 1.05
    X = M / rho
    for _ in range(iters):
        X2 = _mm(X, X)
        X3 = _mm(X2, X)
        X = 1.5 * X - 0.5 * X3
    tr = torch.diagonal(X).sum().item()
    return (d + tr) / 2.0


def _symmetrize(mat):
    m, n = mat.shape
    row = torch.arange(m, device=mat.device).unsqueeze(-1)
    col = torch.arange(n, device=mat.device).unsqueeze(0)
    low = mat * (row >= col)
    strict_low = mat * (row > col)
    return (low + strict_low.transpose(0, 1)).contiguous()


def _effective_tol(atol_b, s, rtol_b, sigma_max_scaled):
    tol = max(atol_b / s, rtol_b * sigma_max_scaled)
    return tol if tol > 0.0 else 0.0


def _rank_single(mat, atol_b, rtol_b, hermitian):
    m, n = mat.shape
    B = _symmetrize(mat) if hermitian else mat.contiguous()
    s = B.abs().max().item()
    if s == 0.0:
        return 0
    Bs = (B / s).contiguous()

    if m <= _SVDVALS_MAX_DIM and n <= _SVDVALS_MAX_DIM:
        S = linalg_svdvals(Bs)
        if S.numel() == 0:
            return 0
        sigma_max = S[0].item()
        tol = _effective_tol(atol_b, s, rtol_b, sigma_max)
        return int((S > tol).sum().item())

    if hermitian:
        sigma_max = _power_iter_specnorm(Bs)
        tol = _effective_tol(atol_b, s, rtol_b, sigma_max)
        eye = torch.eye(Bs.shape[0], device=Bs.device, dtype=torch.float32)
        pos = _count_pos_sign((Bs - tol * eye).contiguous())
        neg = _count_pos_sign((-Bs - tol * eye).contiguous())
        return int(round(pos + neg))

    d = m + n
    H = torch.zeros(d, d, device=Bs.device, dtype=torch.float32)
    H[:m, m:] = Bs
    H[m:, :m] = Bs.transpose(0, 1)
    sigma_max = _power_iter_specnorm(H)
    tol = _effective_tol(atol_b, s, rtol_b, sigma_max)
    eye = torch.eye(d, device=Bs.device, dtype=torch.float32)
    pos = _count_pos_sign((H - tol * eye).contiguous())
    return int(round(pos))


def _flatten_tol(val):
    if isinstance(val, torch.Tensor):
        return val.reshape(-1)
    return val


def _tol_at(val, b):
    if isinstance(val, torch.Tensor):
        return float(val[b].item())
    return float(val)


def _launch_matrix_rank(input, atol_val, rtol_val, hermitian):
    batch_shape = input.shape[:-2]
    m, n = input.shape[-2:]
    flat = input.reshape(-1, m, n)
    nb = flat.shape[0]
    atol_flat = _flatten_tol(atol_val)
    rtol_flat = _flatten_tol(rtol_val)
    ranks = []
    for b in range(nb):
        ranks.append(
            _rank_single(
                flat[b], _tol_at(atol_flat, b), _tol_at(rtol_flat, b), hermitian
            )
        )
    result = torch.tensor(ranks, dtype=torch.int64, device=input.device)
    return result.reshape(batch_shape)


def _needs_negative_tolerance_fixup(atol, rtol):
    if atol is None or rtol is None:
        return False
    if isinstance(atol, torch.Tensor) or isinstance(rtol, torch.Tensor):
        return True
    return float(atol) < 0.0 and float(rtol) < 0.0


def _correct_negative_tolerance_rank(input, result, atol, rtol, hermitian):
    absin = input.abs()
    if hermitian:
        m, n = input.shape[-2], input.shape[-1]
        row = torch.arange(m, device=input.device).unsqueeze(-1)
        col = torch.arange(n, device=input.device).unsqueeze(0)
        absin = absin * (row >= col)
    nonzero = absin.amax(dim=(-2, -1)) > 0
    k = min(input.shape[-2:])
    if isinstance(atol, torch.Tensor) or isinstance(rtol, torch.Tensor):
        neg_pair = (atol < 0) & (rtol < 0)
        return torch.where(neg_pair & nonzero, k, result)
    return torch.where(nonzero, k, result)


def _copy_rank_to_out(input, result, out):
    if out is None:
        raise TypeError("torch.linalg.matrix_rank: out must be a Tensor")
    if out.device != input.device:
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result and input tensors to be on "
            f"the same device, but got result on {out.device} and input on "
            f"{input.device}"
        )
    if not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result to be safely castable from "
            f"Long dtype, but got result with dtype {out.dtype}"
        )
    if out.numel() != 0 and out.shape != result.shape:
        warnings.warn(
            "An output with one or more elements was resized because it had shape "
            f"{tuple(out.shape)}, which does not match the required output shape "
            f"{tuple(result.shape)}.",
            UserWarning,
            stacklevel=3,
        )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def linalg_matrix_rank(input, *, atol=None, rtol=None, hermitian=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_RANK")
    _check_input(input, hermitian)
    output_shape = input.shape[:-2]
    atol_val, rtol_val = _prepare_tolerances(input, atol, rtol)
    if input.numel() == 0:
        return _empty_matrix_rank(input, output_shape)
    result = _launch_matrix_rank(input, atol_val, rtol_val, hermitian)
    if _needs_negative_tolerance_fixup(atol, rtol):
        result = _correct_negative_tolerance_rank(
            input, result, atol_val, rtol_val, hermitian
        )
    return result


def linalg_matrix_rank_tol(input, tol, hermitian=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_RANK_TOL")
    return linalg_matrix_rank(input, atol=tol, rtol=0.0, hermitian=hermitian)


def linalg_matrix_rank_out(input, *, atol=None, rtol=None, hermitian=False, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_RANK_OUT")
    result = linalg_matrix_rank(input, atol=atol, rtol=rtol, hermitian=hermitian)
    return _copy_rank_to_out(input, result, out)


def linalg_matrix_rank_tol_out(input, tol, hermitian=False, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_RANK_TOL_OUT")
    result = linalg_matrix_rank_tol(input, tol, hermitian)
    return _copy_rank_to_out(input, result, out)
