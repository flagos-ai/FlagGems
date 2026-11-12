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

import torch
import triton
import triton.language as tl
from triton import knobs as _knobs
from triton.runtime import driver as _tr_driver

# tl.math.floor exists across Triton 2.x/3.x; tl.floor is the newer core alias.
try:
    _tl_floor = tl.math.floor
except AttributeError:  # pragma: no cover
    _tl_floor = tl.floor

_MAX_DIMS = 6
_MAX_BATCH = 16
# Fusion pays off only while the GPU work is smaller than the launch cost it
# saves; above this many bytes the per-tensor launches are hidden by the GPU.
_FUSE_MAX_BYTES = 1 << 21

# MODE 0: identity (integer dtypes, floor is a no-op)
# MODE 1: floor in fp32 then cast back (fp16 / bf16 / fp32)
# MODE 2: floor in the native dtype (fp64 only, never downcast)
_IDENTITY = 0
_F32 = 1
_NATIVE = 2


@triton.jit
def _fl_body(p, local, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offs = local * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(p + offs, mask=mask)
    if MODE == 0:
        y = x
    elif MODE == 1:
        y = _tl_floor(x.to(tl.float32)).to(x.dtype)
    else:
        y = _tl_floor(x)
    tl.store(p + offs, y, mask=mask)


@triton.jit
def _floor_flat(x_ptr, n_elements, BLOCK: tl.constexpr, MODE: tl.constexpr):
    _fl_body(x_ptr, tl.program_id(0).to(tl.int64), n_elements.to(tl.int64), BLOCK, MODE)


@triton.jit
def _blocks(n, BLOCK: tl.constexpr):
    return (n.to(tl.int64) + (BLOCK - 1)) // BLOCK


# One launch for up to 16 contiguous same-dtype tensors: program ids are split
# across the tensors by cumulative block counts so the per-tensor Triton launch
# cost is paid once. Sizes are do_not_specialize so the compiled kernel is
# reusable for every size combination.
@triton.jit(
    do_not_specialize=[
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
    ]
)
def _floor_fused(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b1 = _blocks(n0, BLOCK)
    b2 = b1 + _blocks(n1, BLOCK)
    b3 = b2 + _blocks(n2, BLOCK)
    b4 = b3 + _blocks(n3, BLOCK)
    b5 = b4 + _blocks(n4, BLOCK)
    b6 = b5 + _blocks(n5, BLOCK)
    b7 = b6 + _blocks(n6, BLOCK)
    b8 = b7 + _blocks(n7, BLOCK)
    b9 = b8 + _blocks(n8, BLOCK)
    b10 = b9 + _blocks(n9, BLOCK)
    b11 = b10 + _blocks(n10, BLOCK)
    b12 = b11 + _blocks(n11, BLOCK)
    b13 = b12 + _blocks(n12, BLOCK)
    b14 = b13 + _blocks(n13, BLOCK)
    b15 = b14 + _blocks(n14, BLOCK)

    if pid < b1:
        _fl_body(x0, pid, n0.to(tl.int64), BLOCK, MODE)
    elif pid < b2:
        _fl_body(x1, pid - b1, n1.to(tl.int64), BLOCK, MODE)
    elif pid < b3:
        _fl_body(x2, pid - b2, n2.to(tl.int64), BLOCK, MODE)
    elif pid < b4:
        _fl_body(x3, pid - b3, n3.to(tl.int64), BLOCK, MODE)
    elif pid < b5:
        _fl_body(x4, pid - b4, n4.to(tl.int64), BLOCK, MODE)
    elif pid < b6:
        _fl_body(x5, pid - b5, n5.to(tl.int64), BLOCK, MODE)
    elif pid < b7:
        _fl_body(x6, pid - b6, n6.to(tl.int64), BLOCK, MODE)
    elif pid < b8:
        _fl_body(x7, pid - b7, n7.to(tl.int64), BLOCK, MODE)
    elif pid < b9:
        _fl_body(x8, pid - b8, n8.to(tl.int64), BLOCK, MODE)
    elif pid < b10:
        _fl_body(x9, pid - b9, n9.to(tl.int64), BLOCK, MODE)
    elif pid < b11:
        _fl_body(x10, pid - b10, n10.to(tl.int64), BLOCK, MODE)
    elif pid < b12:
        _fl_body(x11, pid - b11, n11.to(tl.int64), BLOCK, MODE)
    elif pid < b13:
        _fl_body(x12, pid - b12, n12.to(tl.int64), BLOCK, MODE)
    elif pid < b14:
        _fl_body(x13, pid - b13, n13.to(tl.int64), BLOCK, MODE)
    elif pid < b15:
        _fl_body(x14, pid - b14, n14.to(tl.int64), BLOCK, MODE)
    else:
        _fl_body(x15, pid - b15, n15.to(tl.int64), BLOCK, MODE)


# Same as _floor_fused but for exactly 16 tensors of identical numel, which
# needs only one shared count (17 kernel parameters instead of 34).
@triton.jit(do_not_specialize=["n"])
def _floor_fused_eq(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    n,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = _blocks(n, BLOCK)
    ni = n.to(tl.int64)
    if pid < b:
        _fl_body(x0, pid, ni, BLOCK, MODE)
    elif pid < 2 * b:
        _fl_body(x1, pid - b, ni, BLOCK, MODE)
    elif pid < 3 * b:
        _fl_body(x2, pid - 2 * b, ni, BLOCK, MODE)
    elif pid < 4 * b:
        _fl_body(x3, pid - 3 * b, ni, BLOCK, MODE)
    elif pid < 5 * b:
        _fl_body(x4, pid - 4 * b, ni, BLOCK, MODE)
    elif pid < 6 * b:
        _fl_body(x5, pid - 5 * b, ni, BLOCK, MODE)
    elif pid < 7 * b:
        _fl_body(x6, pid - 6 * b, ni, BLOCK, MODE)
    elif pid < 8 * b:
        _fl_body(x7, pid - 7 * b, ni, BLOCK, MODE)
    elif pid < 9 * b:
        _fl_body(x8, pid - 8 * b, ni, BLOCK, MODE)
    elif pid < 10 * b:
        _fl_body(x9, pid - 9 * b, ni, BLOCK, MODE)
    elif pid < 11 * b:
        _fl_body(x10, pid - 10 * b, ni, BLOCK, MODE)
    elif pid < 12 * b:
        _fl_body(x11, pid - 11 * b, ni, BLOCK, MODE)
    elif pid < 13 * b:
        _fl_body(x12, pid - 12 * b, ni, BLOCK, MODE)
    elif pid < 14 * b:
        _fl_body(x13, pid - 13 * b, ni, BLOCK, MODE)
    elif pid < 15 * b:
        _fl_body(x14, pid - 14 * b, ni, BLOCK, MODE)
    else:
        _fl_body(x15, pid - 15 * b, ni, BLOCK, MODE)


@triton.jit
def _floor_strided(
    x_ptr,
    n_elements,
    SZ0,
    SZ1,
    SZ2,
    SZ3,
    SZ4,
    SZ5,
    ST0,
    ST1,
    ST2,
    ST3,
    ST4,
    ST5,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements

    rem = offs
    i5 = rem % SZ5
    rem = rem // SZ5
    off = i5 * ST5

    i4 = rem % SZ4
    rem = rem // SZ4
    off += i4 * ST4

    i3 = rem % SZ3
    rem = rem // SZ3
    off += i3 * ST3

    i2 = rem % SZ2
    rem = rem // SZ2
    off += i2 * ST2

    i1 = rem % SZ1
    rem = rem // SZ1
    off += i1 * ST1

    off += rem * ST0

    x = tl.load(x_ptr + off, mask=mask)
    if MODE == 0:
        y = x
    elif MODE == 1:
        y = _tl_floor(x.to(tl.float32)).to(x.dtype)
    else:
        y = _tl_floor(x)
    tl.store(x_ptr + off, y, mask=mask)


# ---------------------------------------------------------------------------
# Low-overhead launch path.
#
# A first call per specialization goes through JITFunction.run and yields the
# CompiledKernel; later calls re-enter the very same compiled launcher with
# exactly the argument list that launcher expects (grid, stream, function,
# scratch slots, packed kernel metadata, launch hooks, then every kernel
# parameter in declaration order, constexprs included). Pointers are passed as
# data_ptr() integers, which the launcher accepts directly and which skips its
# per-pointer driver query. Any failure permanently disables the path and the
# standard Triton launch is used instead.
# ---------------------------------------------------------------------------
_FAST = {"ok": True, "ck": {}}


def _make_entry(ck):
    try:
        return (ck, ck.run.launch)
    except Exception:
        _FAST["ok"] = False
        return None


def _direct(ent, stream, gx, gy, gz, vals):
    try:
        ent[1](
            gx,
            gy,
            gz,
            stream,
            ent[0].function,
            None,
            None,
            ent[0].packed_metadata,
            None,
            _knobs.runtime.launch_enter_hook,
            _knobs.runtime.launch_exit_hook,
            *vals,
        )
        return True
    except Exception:
        _FAST["ok"] = False
        return False


def _pick_block(n):
    if n <= 1 << 16:
        return 256, 1
    if n <= 1 << 22:
        return 1024, 4
    return 2048, 4


def _merged_dims(t):
    """[(size, stride), ...] outermost -> innermost (ascending |stride|), contiguous runs merged."""
    dims = [(int(s), int(st)) for s, st in zip(t.shape, t.stride()) if int(s) > 1]
    if not dims:
        return [(1, 0)]
    dims = sorted(dims, key=lambda p: abs(p[1]))
    merged = [dims[0]]
    for sz, st in dims[1:]:
        psz, pst = merged[-1]
        if pst * psz == st and (pst >= 0) == (st >= 0):
            # the merged run is contiguous with the inner (faster) stride
            merged[-1] = (psz * sz, pst)
        else:
            merged.append((sz, st))
    return merged


_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


def _overlaps(dims, n):
    span = 1
    for sz, st in dims:
        span += (sz - 1) * abs(st)
    return span < n


def _launch_strided(t, block, warps, mode):
    n = t.numel()
    dims = _merged_dims(t)
    if len(dims) > _MAX_DIMS:
        for i in range(dims[0][0]):
            _launch_strided(t.select(0, i), block, warps, mode)
        return
    dims = [(1, 0)] * (_MAX_DIMS - len(dims)) + dims
    _floor_strided[((n + block - 1) // block,)](
        t,
        n,
        dims[0][0],
        dims[1][0],
        dims[2][0],
        dims[3][0],
        dims[4][0],
        dims[5][0],
        dims[0][1],
        dims[1][1],
        dims[2][1],
        dims[3][1],
        dims[4][1],
        dims[5][1],
        BLOCK=block,
        MODE=mode,
        num_warps=warps,
    )


def _launch_flat(t, n, ptr, block, warps, mode, dev, stream):
    gx = (n + block - 1) // block
    if _FAST["ok"]:
        key = (
            _floor_flat,
            dev,
            t.dtype,
            block,
            warps,
            mode,
            (n & 15) == 0,
            n == 1,
            n >> 31,
            ptr & 15,
        )
        ent = _FAST["ck"].get(key)
        if ent is None:
            ck = _floor_flat[(gx,)](t, n, BLOCK=block, MODE=mode, num_warps=warps)
            _FAST["ck"][key] = _make_entry(ck)
            return
        if _direct(ent, stream, gx, 1, 1, (ptr, n, block, mode)):
            return
    _floor_flat[(gx,)](t, n, BLOCK=block, MODE=mode, num_warps=warps)


def _launch_fused(tensors, sizes, ptrs, block, warps, mode, n, total, dev, stream):
    pad = _MAX_BATCH - n
    s0 = sizes[0]
    eq = pad == 0 and sizes.count(s0) == _MAX_BATCH
    if eq:
        nblk = _MAX_BATCH * ((s0 + block - 1) // block)
    else:
        nblk = 0
        for s in sizes:
            nblk += (s + block - 1) // block
    kernel = _floor_fused_eq if eq else _floor_fused
    if _FAST["ok"]:
        key = (kernel, dev, tensors[0].dtype, block, warps, mode, total >> 31)
        ent = _FAST["ck"].get(key)
        if ent is not None:
            if eq:
                dvals = ptrs + [s0, block, mode]
            else:
                dvals = ptrs + [0] * pad + sizes + [0] * pad + [block, mode]
            if _direct(ent, stream, nblk, 1, 1, dvals):
                return
        else:
            if eq:
                args = list(tensors) + [s0]
            else:
                args = list(tensors) + [tensors[0]] * pad + sizes + [0] * pad
            ck = kernel[(nblk,)](*args, BLOCK=block, MODE=mode, num_warps=warps)
            _FAST["ck"][key] = _make_entry(ck)
            return
    if eq:
        args = list(tensors) + [s0]
    else:
        args = list(tensors) + [tensors[0]] * pad + sizes + [0] * pad
    kernel[(nblk,)](*args, BLOCK=block, MODE=mode, num_warps=warps)


def _mode_of_checked(dt):
    if dt.is_complex:
        raise NotImplementedError(
            "\"foreach_unary_op_cuda_\" not implemented for '%s'"
            % ("ComplexFloat" if dt is torch.complex64 else "ComplexDouble",)
        )
    if dt is torch.bool:
        raise NotImplementedError("\"floor_cuda\" not implemented for 'Bool'")
    if dt is torch.float64:
        return _NATIVE
    if dt.is_floating_point:
        return _F32
    return _IDENTITY


def _as_list(args, kwargs):
    if kwargs:
        for key in ("self", "tensors", "tensor_list", "inputs", "input", "xs"):
            if key in kwargs:
                return kwargs[key]
        raise TypeError("run() got an unexpected keyword argument")
    if len(args) == 1:
        a = args[0]
        if isinstance(a, (list, tuple)):
            return a
        if torch.is_tensor(a):
            return [a]
        raise TypeError("expected a sequence of tensors")
    return args


def foreach_floor_(*args, **kwargs):
    tensors = _as_list(args, kwargs)
    n = len(tensors)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    if n == 1:
        t = tensors[0]
        if not torch.is_tensor(t):
            raise TypeError(
                "expected Tensor as element 0 in argument 0, but got %s"
                % (type(t).__name__,)
            )
        mode = _mode_of_checked(t.dtype)
        if t.requires_grad and t.is_leaf:
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an in-place operation."
            )
        cnt = t.numel()
        if cnt:
            block, warps = _pick_block(cnt)
            if t.is_contiguous():
                if _FAST["ok"]:
                    dev = torch.cuda.current_device()
                    stream = _tr_driver.active.get_current_stream(dev)
                else:
                    dev = stream = None
                _launch_flat(t, cnt, t.data_ptr(), block, warps, mode, dev, stream)
            else:
                dims = [
                    (int(s), int(st))
                    for s, st in zip(t.shape, t.stride())
                    if int(s) > 1
                ]
                if len(dims) > 1 and _overlaps(dims, cnt):
                    raise RuntimeError(_OVERLAP_MSG)
                _launch_strided(t, block, warps, mode)
        return None

    modes = []
    sizes = []
    ptrs = []
    contigs = []
    mode_cache = {}
    dtype0 = None
    fusable = True
    total = 0
    for idx in range(n):
        t = tensors[idx]
        if not torch.is_tensor(t):
            raise TypeError(
                "expected Tensor as element %d in argument 0, but got %s"
                % (idx, type(t).__name__)
            )
        dt = t.dtype
        mode = mode_cache.get(dt)
        if mode is None:
            mode = _mode_of_checked(dt)
            mode_cache[dt] = mode
        if t.requires_grad and t.is_leaf:
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an in-place operation."
            )
        cnt = t.numel()
        sizes.append(cnt)
        total += cnt
        p = t.data_ptr()
        ptrs.append(p)
        if p & 15:
            fusable = False
        contig = t.is_contiguous()
        contigs.append(contig)
        if cnt == 0:
            modes.append(None)
            fusable = False
            continue
        modes.append(mode)
        if dtype0 is None:
            dtype0 = dt
        elif dt is not dtype0:
            fusable = False
        if not contig:
            fusable = False
            dims = [
                (int(s), int(st)) for s, st in zip(t.shape, t.stride()) if int(s) > 1
            ]
            if len(dims) > 1 and _overlaps(dims, cnt):
                raise RuntimeError(_OVERLAP_MSG)

    if _FAST["ok"]:
        dev = torch.cuda.current_device()
        stream = _tr_driver.active.get_current_stream(dev)
    else:
        dev = stream = None

    if (
        fusable
        and n <= _MAX_BATCH
        and total * tensors[0].element_size() <= _FUSE_MAX_BYTES
    ):
        block, warps = _pick_block(total)
        _launch_fused(
            tensors, sizes, ptrs, block, warps, modes[0], n, total, dev, stream
        )
        return None

    for idx in range(n):
        mode = modes[idx]
        if mode is None:
            continue
        t = tensors[idx]
        block, warps = _pick_block(sizes[idx])
        if contigs[idx]:
            _launch_flat(t, sizes[idx], ptrs[idx], block, warps, mode, dev, stream)
        else:
            _launch_strided(t, block, warps, mode)
    return None
