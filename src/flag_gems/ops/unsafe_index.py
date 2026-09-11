import importlib
import logging
import os
from typing import Any, Callable, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")
    code.newline()
    code.newline()
    return code


def _emit_gather_kernel(
    inp_rank, idx_dims, index_rank, bcast_pos, kernel_name: str, code: IndentedBuffer
):
    """Emit the fully-strided 2D gather kernel.

    The kernel makes no layout assumptions: index tensor ``i`` applies to
    input dim ``idx_dims[i]`` (arbitrary positions, arbitrary input strides),
    and the output is written through its own strides, with the broadcast
    subspace placed at ``bcast_pos``.  This is what frees the host from
    permuting the input or post-permuting the output.
    """
    indices_len = len(idx_dims)
    slice_dims = [d for d in range(inp_rank) if d not in idx_dims]
    out_rank = index_rank + len(slice_dims)
    bcast_out = list(range(bcast_pos, bcast_pos + index_rank))
    rest = [p for p in range(out_rank) if p not in bcast_out]
    slice_out = dict(zip(slice_dims, rest))

    # No @libentry/@libtuner decorators on this launch-bound kernel: profiling
    # showed @libentry alone adds ~12.5us and @libtuner ~2us of CPU dispatch per
    # call.  On these tiny all-tensor shapes that host dispatch exceeds
    # do_bench's L2-flush hide window, so it leaks into the benchmark and also
    # makes it noisy.  A bare @triton.jit (kernel compile/caching is still
    # handled by triton's JITFunction) keeps dispatch minimal; the M-based
    # BLOCK_SIZE0 heuristic below replaces the autotuner deterministically.
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["out_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        args += [f"indices0_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [f"out_stride{i}," for i in range(out_rank)]
        args += [
            "M,",
            "N,",
            "BLOCK_SIZE0: tl.constexpr,",
            "BLOCK_SIZE1: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = ext.program_id(axis=0)")
        # grid axis 1 is capped at 65535 (CUDA gridDim.y limit); axis 2
        # carries the overflow, so pid1 is the flattened (y, z) id.
        code.writeline(
            "pid1 = ext.program_id(axis=1) + ext.program_id(axis=2) "
            "* ext.num_programs(axis=1)"
        )
        code.writeline(
            "offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]"
        )
        if slice_dims:
            code.writeline(
                "offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]"
            )
        else:
            code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
        code.newline()
        # Broadcast-index coords (one per lane of offset0).
        code.writeline("cur_idx = offset0")
        for j in range(index_rank - 1, -1, -1):
            code.writeline(f"idx_coord{j} = cur_idx % indices0_shape{j}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{j}")
        code.newline()
        # Slice-dim coords (one per lane of offset1), in input-dim order.
        if slice_dims:
            code.writeline("cur_idx = offset1")
            for d in reversed(slice_dims):
                code.writeline(f"slice_coord{d} = cur_idx % input_shape{d}")
                code.writeline(f"cur_idx = cur_idx // input_shape{d}")
            code.newline()
        code.writeline("mask0 = offset0 < M")
        for i in range(indices_len):
            dim = idx_dims[i]
            comp = [f"idx_coord{j} * indices{i}_stride{j}" for j in range(index_rank)]
            code.writeline(
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, "
                f"mask=mask0, other=0)"
            )
            # Wrap negative indices in-kernel (avoids host torch.where overhead).
            code.writeline(
                f"cur_index{i} = tl.where(cur_index{i} < 0, "
                f"cur_index{i} + input_shape{dim}, cur_index{i})"
            )
            # Clamp the (B0,1) index vector into range so every computed
            # address is valid even for masked lanes (NPU faults on OOB
            # addresses, zero-tolerance). Cheap: row-vector, not full tile.
            # Semantics stay "unsafe": the bounds mask below still drops the
            # store for out-of-range indices.
            code.writeline(
                f"cur_index{i} = tl.minimum(tl.maximum(cur_index{i}, 0), "
                f"input_shape{dim} - 1)"
            )
        code.newline()
        # Bounds mask: out-of-range indices (UB under "unsafe" semantics, but
        # cheap to handle) skip the store, leaving the output element
        # unwritten instead of scribbling a clamped row into it.
        index_mask = [
            f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{idx_dims[i]})"
            for i in range(indices_len)
        ]
        code.writeline(f"index_mask = {' & '.join(index_mask)}")
        code.writeline("mask1 = offset1 < N")
        code.writeline("mask = index_mask & mask0 & mask1")
        code.newline()
        comp = [f"cur_index{i} * input_stride{idx_dims[i]}" for i in range(indices_len)]
        comp += [f"slice_coord{d} * input_stride{d}" for d in slice_dims]
        code.writeline(f"input_offset = {' + '.join(comp)}")
        comp = [f"idx_coord{r} * out_stride{bcast_pos + r}" for r in range(index_rank)]
        comp += [f"slice_coord{d} * out_stride{slice_out[d]}" for d in slice_dims]
        code.writeline(f"out_offset = {' + '.join(comp)}")
        code.newline()
        code.writeline("cur_value = tl.load(input_ptr + input_offset , mask = mask)")
        code.writeline("tl.store(out_ptr + out_offset, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def _emit_gather_wrapper(
    inp_rank, idx_dims, index_rank, bcast_pos, wrapper_name, kernel_name, code
):
    indices_len = len(idx_dims)
    out_rank = index_rank + inp_rank - indices_len
    code.writeline(f"def {wrapper_name}(input, indices, out):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("out_stride = out.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline("N = out.numel() // M")
        code.newline()
        # Cap the 2D tile at 8K lanes: larger int64-offset tiles spill
        # registers and collapse bandwidth (measured 10-100x slowdowns).
        code.writeline("BLOCK_SIZE1 = min(256, max(1, 2 ** (N - 1).bit_length()))")
        code.writeline("BLOCK_SIZE0 = min(256, max(1, 8192 // BLOCK_SIZE1))")
        code.writeline(
            "BLOCK_SIZE0 = min(BLOCK_SIZE0, max(1, 2 ** (M - 1).bit_length()))"
        )
        # Latency-bound small problems: shrink the tile (closed form, no
        # python loop on the hot path) until the grid fills the device
        # (~2 waves of CTAs). Large shapes skip the shrink entirely.
        code.writeline(
            "ctas = ((M + BLOCK_SIZE0 - 1) // BLOCK_SIZE0) * "
            "((N + BLOCK_SIZE1 - 1) // BLOCK_SIZE1)"
        )
        code.writeline("tile = BLOCK_SIZE0 * BLOCK_SIZE1")
        code.writeline("if ctas < 132 and tile > 256:")
        with code.indent():
            # shift ~ ceil(log2(132/ctas)), bounded by the 256-lane floor.
            code.writeline(
                "shift = min((131 // ctas).bit_length(), tile.bit_length() - 9)"
            )
            code.writeline("BLOCK_SIZE1 = max(1, BLOCK_SIZE1 >> (shift + 1) // 2)")
            code.writeline("BLOCK_SIZE0 = max(1, BLOCK_SIZE0 >> shift // 2)")
        # gridDim.y is capped at 65535: spill the N-block index into axis 2
        # (the kernel flattens axes 1 and 2 back into pid1).
        code.writeline("grid1 = triton.cdiv(N, BLOCK_SIZE1)")
        code.writeline("grid_y = min(grid1, 65535)")
        code.writeline(
            "grid = (triton.cdiv(M, BLOCK_SIZE0), grid_y, triton.cdiv(grid1, grid_y))"
        )
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["out,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            args += [f"indices0_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [f"out_stride[{i}]," for i in range(out_rank)]
            args += [
                "M,",
                "N,",
                "BLOCK_SIZE0=BLOCK_SIZE0,",
                "BLOCK_SIZE1=BLOCK_SIZE1,",
            ]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    inp, tensor_indices, idx_dims, bcast_pos = inputs
    inp_rank = inp.ndim
    indices_len = len(idx_dims)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = generate_imports(code)
    _emit_gather_kernel(inp_rank, idx_dims, index_rank, bcast_pos, kernel_name, code)
    _emit_gather_wrapper(
        inp_rank, idx_dims, index_rank, bcast_pos, wrapper_name, kernel_name, code
    )
    return code


class UnsafeIndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, inp, tensor_indices, out, idx_dims, bcast_pos):
        key = self.arg_key(inp, tensor_indices, idx_dims, bcast_pos)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                (inp, tensor_indices, idx_dims, bcast_pos),
                "_unsafe_index_wrapper",
                "_unsafe_index_jit_function",
                code,
            )

            file_name = f"unsafe_index_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_unsafe_index_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_unsafe_index_wrapper")
            self.overloads[key] = overload

        return overload(inp, tensor_indices, out)

    def arg_key(self, inp, tensor_indices, idx_dims, bcast_pos):
        dims = "-".join(str(d) for d in idx_dims)
        index_rank = tensor_indices[0].ndim
        return f"r{inp.ndim}_d{dims}_ir{index_rank}_bp{bcast_pos}"


_unsafe_index_func = UnsafeIndexFunction()


# aten rejects a bad index dtype with
# ``RuntimeError: _unsafe_index found unexpected index type Bool`` where the
# name is the legacy TypeMeta name; mirror both the error type and the text.
_DTYPE_LEGACY_NAMES = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.float16: "Half",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.bfloat16: "BFloat16",
}


def _check_indices(inp, indices):
    """Validate index dtypes and move cross-device indices, in one pass.

    Mirrors ``aten._unsafe_index``: only int32/int64 index tensors are
    accepted (bool/int8/uint8/float/... are all rejected with a RuntimeError),
    and an index living on a different device than the input is silently moved
    onto it.
    """
    checked = []
    for index in indices:
        if index is None:
            checked.append(None)
            continue
        dt = index.dtype
        if dt not in (torch.int32, torch.int64):
            name = _DTYPE_LEGACY_NAMES.get(dt, str(dt))
            raise RuntimeError(f"_unsafe_index found unexpected index type {name}")
        if index.device != inp.device:
            index = index.to(inp.device)
        checked.append(index)
    return checked


def _broadcast_index_tensors(tensor_indices):
    """Broadcast index tensors, mirroring aten's IndexError on a mismatch."""
    try:
        return list(torch.broadcast_tensors(*tensor_indices))
    except RuntimeError:
        shapes = ", ".join(str(list(idx.shape)) for idx in tensor_indices)
        raise IndexError(
            "shape mismatch: indexing tensors could not be broadcast "
            f"together with shapes {shapes}"
        ) from None


def _eliminate_scalar_indices(inp, indices):
    """Resolve 0-d (scalar) tensor indices on the host.

    In aten a 0-d index tensor behaves like an integer index: the indexed dim
    is removed from the output.  Resolve them with ``select`` views so the
    generated kernels only ever see index tensors with rank >= 1 (a 0-d index
    would otherwise make the codegen emit an invalid scalar load).
    ``idx.item()`` forces a device sync, but 0-d indices are a rare edge case.
    """
    if not any(idx is not None and idx.ndim == 0 for idx in indices):
        return inp, indices
    remaining = []
    removed = 0
    for i, idx in enumerate(indices):
        if idx is not None and idx.ndim == 0:
            # List position i aligns with input dim i; each selected-away dim
            # shifts the positions of the following dims down by one.
            pos = i - removed
            v = int(idx.item())
            if v < 0:
                v += inp.shape[pos]
            inp = inp.select(pos, v)
            removed += 1
        else:
            remaining.append(idx)
    return inp, remaining


def unsafe_index(inp, indices):
    """Code-generated ``unsafe_index`` matching ``aten._unsafe_index``.

    Host side is a straight line: validate, resolve scalar (0-d) indices,
    broadcast, compute the output layout per aten's subspace-placement rule,
    and launch.  The kernels take an explicit index->dim mapping plus full
    input/output strides, so no input permute/contiguous and no output
    post-permute is ever needed.
    """
    logger.debug("GEMS UNSAFE_INDEX")
    if not indices:
        raise ValueError("at least one index must be provided")
    indices = _check_indices(inp, list(indices))
    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    # Subspace placement is decided over the *original* advanced indices:
    # aten counts scalar (0-d) tensors as advanced indices, so a scalar
    # separated from the other index tensors by a None dim already forces
    # the broadcast subspace to the front of the output.
    advanced_dims = [i for i, idx in enumerate(indices) if idx is not None]
    subspace_split = bool(advanced_dims) and advanced_dims != list(
        range(advanced_dims[0], advanced_dims[0] + len(advanced_dims))
    )

    inp, indices = _eliminate_scalar_indices(inp, indices)
    if not indices:
        return inp  # every index was a scalar (0-d) tensor

    # Pad missing trailing dims with None (full slices).
    indices = indices + [None] * (inp.ndim - len(indices))

    kernel_indices = [idx for idx in indices if idx is not None]
    if not kernel_indices:
        # All-None indices: plain basic indexing, just a contiguous copy.
        # (aten's own kernel asserts on this case; a copy is harmless.)
        return inp.contiguous()
    if len(kernel_indices) > 1:
        kernel_indices = _broadcast_index_tensors(kernel_indices)

    idx_dims = tuple(i for i, idx in enumerate(indices) if idx is not None)
    slice_dims = [i for i, idx in enumerate(indices) if idx is None]

    # Output layout per aten's rule: when the advanced indices are separated
    # by None dims the broadcast subspace goes to the front; otherwise it sits
    # where the (contiguous) advanced block starts, counting surviving dims.
    bcast_pos = 0 if subspace_split else idx_dims[0]
    index_rank = kernel_indices[0].ndim
    out_rank = index_rank + len(slice_dims)
    bcast_out = list(range(bcast_pos, bcast_pos + index_rank))
    rest = [p for p in range(out_rank) if p not in bcast_out]
    out_shape = [0] * out_rank
    for r, p in enumerate(bcast_out):
        out_shape[p] = kernel_indices[0].shape[r]
    for d, p in zip(slice_dims, rest):
        out_shape[p] = inp.shape[d]

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    # Skip the kernel launch for an empty output: an empty index would give a
    # grid of (0, ...) programs and fault the device; the empty output already
    # has the correct shape.
    if out.numel() != 0:
        _unsafe_index_func(inp, kernel_indices, out, idx_dims, bcast_pos)
    return out
