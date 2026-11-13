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

"""Triton implementation of ``aten::_sparse_broadcast_to_copy`` for sparse COO.

The copy variant shares the broadcasting semantics of
``aten::_sparse_broadcast_to`` — replicating stored entries over the newly
prepended and expanded sparse dimensions while copying payload values bitwise
through the target dense strides — but always materialises a fresh tensor and
additionally provides the ``out=`` form.

The KernelGen-qualified ``sparse_broadcast_to`` kernel already builds fresh
indices and values (it never aliases the input storage), so this module
reuses it directly and only adds the copy-specific ``out=`` handling.

Enflame (gcu400) adaptations:

- torch_gcu ships no SparseCOO TensorImpl: sparse COO tensors materialize on
  the CPU (``_dimI.ensure_sparse_coo_gcu_fallback``), so ``self._indices()``
  and ``self._values()`` of a test- or factory-built operand are CPU tensors
  and ``self.device`` reports cpu. The kernel still has to run on the GCU
  (host functions must not fall back to aten compute), so the components are
  moved to the current GCU device for the launch. The copy variant's accuracy
  contract compares the components bitwise against the native reference
  output, which on this platform carries CPU components, so the freshly
  computed components are copied back to the CPU whenever the operand's own
  components were on the CPU (device parity with the native operator; an
  operand whose components already sit on the device keeps device
  components). Same pattern as sparse_broadcast_to.py / sspaddmm.py.
- The index operands ride the int64 layout torch_gcu actually uses: with
  ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` enabled the int64 tensors are true
  8-byte elements and the kernel loads/stores them directly; with the env
  off torch_gcu applies its Long->Int replacement (4-byte slots, dtype
  metadata unchanged) and the kernel is fed int32 views of the same buffers
  so every access hits the slot width torch will read back (see P17 in
  compiler_problems/). Values stay inside the int32 safe domain (nnz_new and
  every target extent below 2**31); outside it the int64 arm is kept as-is.
  Same dual-branch store as kthvalue.py.
- The identity case (F == 1) is a pure metadata/view path with no kernel
  launch: the operand's own components pass through unchanged, exactly like
  the native operator that aliases them.
"""

import logging
import os

import torch
import triton

from flag_gems.ops.sparse_broadcast_to import (
    _BLOCK_E,
    _MAXD,
    _MAXDN,
    _as_size_tuple,
    _next_pow2,
    _out_coalesced,
    _sb_fused_kernel,
    _sb_idx_kernel,
    _sb_val_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _gcu_int64_layout():
    # torch_gcu reads an int64 tensor with a true 8-byte element stride only
    # when TORCH_GCU_ENABLE_INT64_AND_UINT64 selects the int64 layout; with
    # the env off it applies the Long->Int replacement and stores 4-byte
    # slots while the dtype metadata still says int64 (P17). The kernel must
    # match the slot width or torch reads back interleaved halves. Accepted
    # values mirror torch_gcu's parse (measured 2026-11-12, kthvalue.py):
    # "1"/"true"/nonzero numerics enable int64; unset/"0"/"false"/"TRUE"/
    # "on"/"yes" keep the replacement. Checked per launch because torch_gcu
    # flips the layout inside a process via AllowINT64AndUINT64Ctx.
    val = os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64")
    if val is None:
        return False
    if val == "true":
        return True
    try:
        return int(val) != 0
    except ValueError:
        return False


def _index_arg(buffer, int64_layout):
    """Hand the kernel the index buffer at the width torch will read it."""
    if int64_layout:
        return buffer
    return buffer.view(torch.int32)


def sparse_broadcast_to_copy(self, size, out=None):
    logger.debug("GEMS SPARSE BROADCAST TO COPY")
    result = _sparse_broadcast_to_copy_impl(self, size)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _sparse_broadcast_to_copy_impl(self, size):
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise NotImplementedError(
            "aten::_sparse_broadcast_to: only sparse COO operands are supported"
        )

    tgt = _as_size_tuple(size)

    sp = self.sparse_dim()
    dn = self.dense_dim()
    D = len(tgt)
    if D < sp + dn:
        raise RuntimeError(
            "the requested broadcast shape has fewer dimensions than the input"
        )

    src_shape = self.size()
    sp_new = D - dn
    lead = sp_new - sp

    for i in range(sp):
        s = src_shape[i]
        t = tgt[lead + i]
        if s != t and s != 1:
            raise RuntimeError(
                f"The input's length {s} at dimension {i} does not broadcast over "
                f"the requested shape of length {t} at dimension {lead + i}"
            )
    for k in range(dn):
        s = src_shape[sp + k]
        t = tgt[sp_new + k]
        if s != t and s != 1:
            raise RuntimeError(
                f"The input's length {s} at dimension {sp + k} does not broadcast "
                f"over the requested shape of length {t} at dimension {sp_new + k}"
            )

    # ---- expansion factors, mixed-radix strides, source rows ---------------
    ext = [1] * sp_new
    srcrow = [-1] * sp_new
    for d in range(lead):
        ext[d] = tgt[d]
    for i in range(sp):
        d = lead + i
        if src_shape[i] == 1 and tgt[d] > 1:
            ext[d] = tgt[d]
        else:
            srcrow[d] = i
    strd = [1] * sp_new
    acc = 1
    for d in range(sp_new - 1, -1, -1):
        strd[d] = acc
        acc *= ext[d]
    F = acc

    # ---- output geometry ---------------------------------------------------
    tgt_dense = [tgt[sp_new + k] for k in range(dn)]
    src_dense = [src_shape[sp + k] for k in range(dn)]
    V = 1
    for en in tgt_dense:
        V *= en

    nnz = self._nnz()
    nnz_new = nnz * F

    if F == 1:
        # Pure view path with no kernel launch; the operand's own components
        # pass through unchanged (the native operator aliases them), so the
        # identity case keeps the platform's component device as given.
        return torch.sparse_coo_tensor(
            self._indices(),
            self._values().expand((nnz,) + tuple(tgt_dense)),
            tuple(tgt),
            is_coalesced=self.is_coalesced(),
        )

    # ---- enflame: resolve the compute device from the components ----------
    # GCU400 cannot host a sparse COO TensorImpl: the input's components
    # materialize on the CPU (see _dimI.ensure_sparse_coo_gcu_fallback) even
    # when the data came from the device. Compute on the GCU (never fall
    # back to host aten) and mirror the component device of the native
    # operator's output: CPU components in, CPU components out.
    src_idx = self._indices()
    src_val = self._values()
    if src_val.device.type != "cpu":
        compute_device = src_val.device
    elif src_idx.device.type != "cpu":
        compute_device = src_idx.device
    else:
        compute_device = torch.device(f"gcu:{torch_device_fn.current_device()}")
    result_on_cpu = src_val.device.type == "cpu" and src_idx.device.type == "cpu"
    if src_idx.device != compute_device:
        src_idx = src_idx.to(compute_device)
    if src_val.device != compute_device:
        src_val = src_val.to(compute_device)

    out_idx = torch.empty((sp_new, nnz_new), dtype=torch.int64, device=compute_device)
    out_val = torch.empty(
        (nnz_new,) + tuple(tgt_dense), dtype=self.dtype, device=compute_device
    )

    if nnz_new > 0 and (sp_new > 0 or V > 0):
        stride0 = src_val.stride(0) if src_val.dim() > 0 else 1

        sstrd = [0] * dn
        tstrd = [0] * dn
        acc = 1
        for k in range(dn - 1, -1, -1):
            tstrd[k] = acc
            acc *= tgt_dense[k]
        vstrides = src_val.stride()
        for k in range(dn):
            sstrd[k] = vstrides[k + 1] if src_dense[k] > 1 else 0

        try:
            no_other = torch.finfo(self.dtype).bits < 16
        except (TypeError, ValueError):
            no_other = False

        # The index buffers ride the slot width torch will read them back
        # with: true int64 elements under the int64 layout, or their int32
        # word view under torch_gcu's Long->Int replacement (P17). The
        # int32 view is only exact inside the int32 safe domain; outside it
        # keep int64 (correct whenever the int64 layout is active).
        safe32 = nnz_new < 2**31 and (not tgt or max(tgt) < 2**31)
        int64_layout = _gcu_int64_layout() or not safe32

        if sp_new <= _MAXD.value and dn <= _MAXDN.value:
            # A wide payload with few entries (the dense-expansion cases) wants
            # a small entry block and a narrow payload block: the measured
            # per-case optima were entry 128-256 / payload 8-32 there, while
            # the replication cases want the full 256.
            if dn and V > 1 and nnz_new < 65536:
                block_e = min(128, max(16, _next_pow2(nnz_new)))
                block_v = min(32, _next_pow2(V))
            else:
                block_e = _BLOCK_E
                block_v = min(_next_pow2(max(V, 1)), 128)
            n_v = triton.cdiv(V, block_v) if V > 0 else 1
            grid = (triton.cdiv(nnz_new, block_e), n_v)
            # F == 1 short-circuits the decode entirely (FL2=0, FMASK=0); a
            # power-of-two F turns it into a shift and a mask instead of an
            # int64 divide.
            pow2 = (F & (F - 1)) == 0
            with torch_device_fn.device(compute_device):
                _sb_fused_kernel[grid](
                    _index_arg(src_idx, int64_layout),
                    src_val,
                    _index_arg(out_idx, int64_layout),
                    out_val,
                    nnz,
                    nnz_new,
                    F.bit_length() - 1,
                    F - 1,
                    F,
                    V,
                    stride0,
                    *(ext + [1] * (_MAXD.value - sp_new)),
                    *(srcrow + [-1] * (_MAXD.value - sp_new)),
                    *(tstrd + [1] * (_MAXDN.value - dn)),
                    *(tgt_dense + [1] * (_MAXDN.value - dn)),
                    *(sstrd + [0] * (_MAXDN.value - dn)),
                    SP_NEW=sp_new,
                    SP_BLOCK=_next_pow2(max(sp_new, 1)),
                    DN=dn,
                    BLOCK_E=block_e,
                    BLOCK_V=block_v,
                    POW2=pow2,
                    NO_OTHER=no_other,
                )
        else:
            _fallback_launch(
                src_idx,
                int64_layout,
                out_idx,
                out_val,
                src_val,
                stride0,
                ext,
                strd,
                srcrow,
                tgt_dense,
                tstrd,
                sstrd,
                nnz,
                nnz_new,
                F,
                V,
                sp_new,
                dn,
                compute_device,
                no_other,
            )

    if result_on_cpu:
        # The native operator materializes its output on the same device as
        # the operand's components; the copy variant's accuracy contract
        # compares the components bitwise, so land them where the reference
        # expects them.
        out_idx = out_idx.cpu()
        out_val = out_val.cpu()

    out_flag = _out_coalesced(self.is_coalesced(), sp, dn, sp_new, ext)
    return torch.sparse_coo_tensor(out_idx, out_val, tuple(tgt), is_coalesced=out_flag)


def _fallback_launch(
    src_idx,
    int64_layout,
    out_idx,
    out_val,
    src_val,
    stride0,
    ext,
    strd,
    srcrow,
    tgt_dense,
    tstrd,
    sstrd,
    nnz,
    nnz_new,
    F,
    V,
    sp_new,
    dn,
    device,
    no_other,
):
    """Correct-but-slower path for shapes outside the scalar-argument kernel."""
    # int32 metadata whenever the int64 layout is not active: torch_gcu's
    # Long->Int replacement stores those device tensors as 4-byte slots and
    # an int64 load would read interleaved halves (P17).
    meta_dtype = torch.int64 if int64_layout else torch.int32
    meta = torch.tensor(ext + strd + srcrow, dtype=meta_dtype, device=device)
    grid = (triton.cdiv(nnz_new, _BLOCK_E),)
    with torch_device_fn.device(device):
        _sb_idx_kernel[grid](
            _index_arg(src_idx, int64_layout),
            _index_arg(out_idx, int64_layout),
            meta,
            nnz_new,
            nnz,
            F,
            sp_new,
            SP_BLOCK=_next_pow2(sp_new),
            BLOCK_E=_BLOCK_E,
        )
    if V > 0:
        dmeta = []
        for k in range(dn):
            dmeta += [tstrd[k], tgt_dense[k], sstrd[k]]
        dmeta_t = torch.tensor(dmeta, dtype=meta_dtype, device=device)
        blk_v = min(_next_pow2(V), 128)
        grid = (triton.cdiv(nnz_new, _BLOCK_E), triton.cdiv(V, blk_v))
        with torch_device_fn.device(device):
            _sb_val_kernel[grid](
                src_val,
                out_val,
                dmeta_t,
                nnz_new,
                F,
                V,
                stride0,
                DN=dn,
                BLOCK_E=_BLOCK_E,
                BLOCK_V=blk_v,
                NO_OTHER=no_other,
            )
