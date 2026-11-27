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
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safe domain: the masked tail tile evaluates
# tile_id * BLOCK + arange for masked lanes as well, reaching up to
# tile_count * BLOCK - 1; keep n_floats <= 2^31 - 2^17
# so flat int32 offsets cannot wrap negative on the tail tile.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)


@libentry()
@triton.jit(do_not_specialize=["n_floats"])
def _conj_copy_flat_kernel(
    in_ptr,
    out_ptr,
    n_floats,
    BLOCK: tl.constexpr,
):
    # Fixed grid + in-kernel stride walk: grid decouples from problem size,
    # so large shapes do not spawn a huge CTA count.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    tile_count = tl.cdiv(n_floats, BLOCK)
    for tile_id in tl.range(pid, tile_count, num_pids):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_floats
        values = tl.load(in_ptr + offsets, mask=mask, other=0)
        # Interleaved layout (real, imag, ...): odd offsets hold the
        # imaginary parts; bit-and replaces the modulo.
        is_imaginary = (offsets & 1) == 1
        result = tl.where(is_imaginary, -values, values)
        tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elems"])
def _conj_storage_copy_kernel(
    real_ptr,
    imag_ptr,
    out_ptr,
    n_elems,
    BLOCK: tl.constexpr,
):
    # Used only for inputs that carry the conjugate bit, whose physical storage
    # already holds the conjugated values. Both Tensor.resolve_conj() and
    # Tensor.copy_() keep the bit set on GCU, so any tensor derived from such an
    # input still stores the same physical bytes; kernel loads read that
    # physical storage. The parts are therefore copied verbatim (no negation)
    # and the fresh output buffer, which has no conjugate bit, exposes them as
    # the conjugated copy.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    tile_count = tl.cdiv(n_elems, BLOCK)
    for tile_id in tl.range(pid, tile_count, num_pids):
        indices = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = indices < n_elems
        real_values = tl.load(real_ptr + indices, mask=mask, other=0)
        imag_values = tl.load(imag_ptr + indices, mask=mask, other=0)
        tl.store(out_ptr + 2 * indices, real_values, mask=mask)
        tl.store(out_ptr + 2 * indices + 1, imag_values, mask=mask)


# GCU400 retune (2026-11-16, do_bench(median) probe sweep over the official
# core shapes): conj-copy is memory-bound (copy floor 3.06 TB/s at 512M
# complex; native torch._conj_copy sits exactly on that floor), so only the
# launch shape matters. Measured per band on the complex64 float stream:
#   tiny (n_floats <= 16K): multi-CTA B1024/nw4 hits the launch floor
#     (3.4us vs 3.7us single-CTA B8192 at 8K floats);
#   16K < n_floats <= 2^22: B8192/nw4 leads B32768/nw2 by ~5-8%
#     (5.7 vs 6.2us @512K floats, 10.9 vs 11.5us @2M floats);
#   n_floats > 2^22: B32768/nw2/g48 rides the copy floor
#     (2710us vs 3727us for the old B4096/nw4/g192 at 2^30 floats;
#     16M-complex band 90.4us vs 124.2us; native 2708/91.1us).
#   n_floats > 2^26: B32768/nw4/g384 (16 CTAs per sip) wins once the working
#     set is far past L2 (5130us vs 5351us at 2^30 floats, native 4023us);
#     at 2^24 floats the same wide grid regresses (93.4us -> 109.6us), hence
#     the separate band.
# Mask-free EVEN tiles, negate-by-multiply, xor-on-sign-bit and num_stages
# 2/3 were all probed and none beat the masked where form (WITHIN-NOISE or
# worse; xor also risks the P18 bitcast-chain pathology), so the single
# masked kernel above keeps the previous semantics.
def _choose_launch(n_floats):
    """Return (BLOCK, num_warps, grid_multiplier) for the flat stream length.

    Below 4M floats one CTA per sip is enough at the launch floor; the
    4M..64M-float band (the official mid-size shapes) needs two CTAs per sip to
    keep the copy floor, and only the widest band (far past L2, multGB working
    set) gains from 8 CTAs per sip with 4 warps.
    """
    if n_floats <= (1 << 22):
        return 8192, 4, 1
    if n_floats <= (1 << 26):
        # 2^25-float band (the official 4096x4096 / 64x512x512 cases): a 256KB
        # tile over 48 CTAs rides the copy floor, 94.1us vs 103.7us for the
        # previous B32768/g1 (native 94.5us), i.e. 0.91x -> 1.00x.
        return 65536, 2, 2
    if n_floats <= (1 << 27):
        return 32768, 2, 2
    return 32768, 4, 8


def _float_dtype(x: torch.Tensor) -> torch.dtype:
    # complex32 / complex64 / complex128 each map to their real component.
    if x.dtype == torch.complex32:
        return torch.float16
    if x.dtype == torch.complex64:
        return torch.float32
    if x.dtype == torch.complex128:
        return torch.float64
    raise RuntimeError("conj only supports complex tensors")


def _input_real_stream(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret the (contiguous) complex input as a flat real stream.

    On GCU, resolve_conj() can still leave an unresolved conj view and
    PyTorch then rejects .view(real_dtype); rebuild a plain complex tensor
    from the logical real/imag values in that case so the reinterpret
    always goes through.
    """
    float_dtype = _float_dtype(tensor)
    try:
        return tensor.reshape(-1).view(float_dtype)
    except RuntimeError:
        rebuilt = torch.complex(tensor.real.contiguous(), tensor.imag.contiguous())
        return rebuilt.reshape(-1).view(float_dtype)


def _output_real_stream(tensor: torch.Tensor) -> torch.Tensor:
    # Output side: direct reinterpret only. The output buffer is always a
    # freshly allocated (or user-supplied contiguous) tensor, so a rebuild
    # fallback would write to a detached copy; a hard error is the correct
    # behavior here.
    return tensor.reshape(-1).view(_float_dtype(tensor))


def _launch_conj_flat(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Conjugate-write a contiguous complex tensor into dst as flat streams."""
    n_floats = src.numel() * 2
    if n_floats == 0:
        return
    block, num_warps, grid_mult = _choose_launch(n_floats)
    grid_size = min(triton.cdiv(n_floats, block), NUM_SIPS * grid_mult)
    in_flat = _input_real_stream(src)
    out_flat = _output_real_stream(dst)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # rsqrt.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(src.device):
                _conj_copy_flat_kernel[(grid_size,)](
                    in_flat,
                    out_flat,
                    n_floats,
                    BLOCK=block,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "conj flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _launch_conj_storage(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy the physical storage of a conjugate-bit complex input into dst.

    The physical storage already holds the conjugated values, so the flat
    interleaved negate kernel cannot be used (it would negate a second time).
    The real / imaginary float views are copied verbatim into the freshly
    allocated output, which carries no conjugate bit.
    """
    n_elems = src.numel()
    if n_elems == 0:
        return
    real_flat = src.real.contiguous().reshape(-1)
    imag_flat = src.imag.contiguous().reshape(-1)
    out_flat = _output_real_stream(dst)
    block, num_warps, grid_mult = _choose_launch(n_elems)
    grid_size = min(triton.cdiv(n_elems, block), NUM_SIPS * grid_mult)
    with torch_device_fn.device(src.device):
        _conj_storage_copy_kernel[(grid_size,)](
            real_flat,
            imag_flat,
            out_flat,
            n_elems,
            BLOCK=block,
            num_warps=num_warps,
        )


def _flat_path_ok(tensor: torch.Tensor) -> bool:
    return (
        tensor.is_contiguous()
        and tensor.device.type == "gcu"
        and tensor.numel() * 2 < _INT32_SAFE_LIMIT
    )


def _conj_copy(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _CONJ_COPY")
    if not input.is_complex():
        raise RuntimeError("_conj_copy only supports complex tensors")

    # A pre-existing conjugate bit means the physical storage already holds the
    # conjugated values, so the interleaved negate kernel would negate twice.
    # GCU Tensor.resolve_conj() keeps the bit set, so the two cases are
    # dispatched explicitly instead of relying on the bit being resolved.
    has_conj_bit = input.is_conj()

    src = input if input.is_contiguous() else input.contiguous()
    if not _flat_path_ok(src):
        from flag_gems.ops._conj_copy import _conj_copy as _public

        return _public(src)

    output = torch.empty_like(src)
    if has_conj_bit:
        _launch_conj_storage(src, output)
    else:
        _launch_conj_flat(src, output)
    return output


def _conj_copy_out(input: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _CONJ_COPY_OUT")
    if not input.is_complex():
        raise RuntimeError("_conj_copy only supports complex tensors")
    if out.dtype != input.dtype:
        raise RuntimeError("_conj_copy.out: dtype of input and output must match.")
    if out.device != input.device:
        raise RuntimeError(
            "_conj_copy.out: input and output must be on the same device."
        )
    if out.numel() != input.numel():
        raise RuntimeError(
            "_conj_copy.out: input and output must have the same number of elements."
        )

    # Same conjugate-bit dispatch as _conj_copy: the bit is not resolved by
    # GCU, so an input that already carries it needs the logical-part path.
    has_conj_bit = input.is_conj()

    src = input if input.is_contiguous() else input.contiguous()
    dst = out if out.is_contiguous() else out.contiguous()
    if not (_flat_path_ok(src) and _flat_path_ok(dst)):
        from flag_gems.ops._conj_copy import _conj_copy_out as _public

        return _public(input, out=out)

    if has_conj_bit:
        _launch_conj_storage(src, dst)
    else:
        _launch_conj_flat(src, dst)

    if dst.data_ptr() != out.data_ptr():
        out.copy_(dst)
    return out
