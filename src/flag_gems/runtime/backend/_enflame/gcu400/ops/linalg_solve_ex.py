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

"""gcu400 linalg_solve_ex: shape-dispatched fast paths over the ATen fallback.

Background: the generic Triton kernel keeps the LU/X workspaces in global
memory and interleaves vector stores with scalar loads inside one program; on
gcu400 the scalar load can observe the pre-store value even with explicit
barriers (fp32 residual 1.5e-2 for cond~4 systems at n=16/32/64), so the
public op was switched to a redispatch of ``aten::linalg_solve_ex``
(CompositeImplicitAutograd).  That decomposition lands in the FlagGems
``_linalg_solve_ex`` fused Triton kernel, whose per-batch elimination walks a
long serial chain of dependent global loads (~200us for n=16, ~0.45x of
native).  A register-tile rewrite (modeled on the lu_factor_ex register
kernel) was measured at ~148us: gcu400 charges ~0.4-0.6us per cross-lane
extraction/broadcast, so that whole family of kernels floors above the
5-kernel native path.

Measured crossover on gcu400 (do_bench kernel mode, fp32):
- n <= 4: the FlagGems fused Triton kernel wins (19us vs 24-29us for native
  and the vendor composition below) thanks to its single tiny launch, so the
  ATen composite redispatch (which routes to that kernel) is kept.
- n >= 5: the vendor composition wins.  The torch_gcu leaf kernels
  ``aten::linalg_lu_factor_ex.out`` and ``aten::linalg_lu_solve.out`` are
  captured before ``use_gems`` installs the FlagGems registrations and called
  directly with pre-allocated outputs, skipping the ATen composite, the
  dispatcher hops and the private-op host logic of the native path.  This is
  6-19% faster than native ``torch.linalg.solve_ex`` wall time on the
  benchmark shapes while producing bit-identical vendor kernels and the same
  (result, info) semantics.

Everything else (left=False, vector right-hand sides, broadcast batches,
other dtypes, empty inputs, CPU tensors) keeps the exact redispatch behavior
of the previous implementation.
"""

import torch

import flag_gems
from flag_gems import runtime


__all__ = ["linalg_solve_ex"]

_COMPOSITE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)

# Captured at import time, before ``use_gems`` installs the FlagGems kernels,
# so the vendor leaf kernels stay reachable (same approach as the complex path
# of _linalg_solve_ex / linalg_matrix_sqrth).  PyTorch builds without
# get_kernel keep importing FlagGems normally and simply lose this fast path.
_CAN_CAPTURE_NATIVE_KERNELS = hasattr(torch.library, "get_kernel")

if _CAN_CAPTURE_NATIVE_KERNELS:
    _DEVICE_KEYSET = torch._C.DispatchKeySet(
        getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
    )
    try:
        _NATIVE_LU_FACTOR_EX_OUT = torch.library.get_kernel(
            torch.ops.aten.linalg_lu_factor_ex.out, runtime.device.dispatch_key
        )
        _NATIVE_LINALG_LU_SOLVE_OUT = torch.library.get_kernel(
            torch.ops.aten.linalg_lu_solve.out, runtime.device.dispatch_key
        )
    except Exception:  # pragma: no cover - vendor kernels always exist today
        _CAN_CAPTURE_NATIVE_KERNELS = False

# Below this size the FlagGems fused Triton kernel (reached through the ATen
# composite redispatch) is faster than the two vendor kernel launches.
_VENDOR_SOLVE_MIN_N = 5


def _can_use_vendor_solve(A, B, left):
    """Whether the captured vendor lu_factor_ex + lu_solve pair covers this
    call: plain fp32 ``AX = B`` with matrix right-hand sides, matching batch
    shapes and n large enough for the vendor pair to win."""
    if not _CAN_CAPTURE_NATIVE_KERNELS or not left:
        return False
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        return False
    if A.ndim < 2 or B.ndim < 2:
        return False
    if A.shape[-1] != A.shape[-2]:
        return False
    n = A.shape[-1]
    if n < _VENDOR_SOLVE_MIN_N:
        return False
    if B.shape[-2] != n:
        return False
    if tuple(A.shape[:-2]) != tuple(B.shape[:-2]):
        return False
    if A.numel() == 0 or B.numel() == 0:
        return False
    if A.device.type != runtime.device.name:
        return False
    return True


def _raise_singular(info):
    index = int(flag_gems.argmax((info != 0).reshape(-1).to(torch.int32)))
    raise torch.linalg.LinAlgError(
        f"torch.linalg.solve_ex: (Batch element {index}): The solver failed "
        "because the input matrix is singular."
    )


def _vendor_solve_ex(A, B, check_errors):
    """Factor with the vendor getrf and solve with the vendor lu_solve.

    Both kernels are the leaf ``.out`` overloads that the native ATen path
    itself bottoms out in, so the numerical results and the ``info`` tensor
    match ``torch.linalg.solve_ex`` exactly; only the host-side composition
    is leaner.
    """
    n = A.shape[-1]
    batch_shape = A.shape[:-2]

    LU = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    pivots = torch.empty(batch_shape + (n,), dtype=torch.int32, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
    _NATIVE_LU_FACTOR_EX_OUT.call_boxed(
        _DEVICE_KEYSET,
        A,
        pivot=True,
        check_errors=False,
        LU=LU,
        pivots=pivots,
        info=info,
    )

    X = torch.empty(B.shape, dtype=A.dtype, device=A.device)
    _NATIVE_LINALG_LU_SOLVE_OUT.call_boxed(
        _DEVICE_KEYSET, LU, pivots, B, left=True, adjoint=False, out=X
    )

    if check_errors and flag_gems.any(info != 0):
        _raise_singular(info)

    return X, info


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    """Solve AX = B, returning (result, info); see the module docstring."""
    if _can_use_vendor_solve(A, B, left):
        return _vendor_solve_ex(A, B, check_errors)

    return torch.ops.aten.linalg_solve_ex.default.redispatch(
        _COMPOSITE_KEYSET, A, B, left=left, check_errors=check_errors
    )
