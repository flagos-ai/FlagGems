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
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")
    code.newline()
    code.newline()
    return code


def _generate_segment_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    """Emit a 1D contiguous-segment kernel for multi-index (indices_len < inp_rank).

    Each program handles ONE index position (scalar pid0) and copies a
    contiguous post block -> 1D load/store -> large DMA.  This avoids the
    2D-broadcast pathology of the all-tensor kernel on large post dims.
    Negative indices are wrapped in-kernel (scalar tl.where, safe on TBE).
    """
    code.writeline("@libentry()")
    code.writeline("@triton.autotune(")
    with code.indent():
        code.writeline(
            "configs=[triton.Config({'BLOCK_SIZE1': b}) for b in "
            "(64,128,256,512,1024,2048,4096)],"
        )
        code.writeline('key=["N"],')
    code.writeline(")")
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
        args += ["M,", "N,", "BLOCK_SIZE1: tl.constexpr,"]
        code.writelines(args)
    code.writeline("):")
    with code.indent():
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("pid1 = ext.program_id(axis=1)")
        # decompose pid0 into index coords (using indices0_shape)
        code.writeline("cur_idx = pid0")
        for j in range(index_rank - 1, -1, -1):
            code.writeline(f"indices_idx{j} = cur_idx % indices0_shape{j}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{j}")
        # load k scalar indices, wrap negatives, accumulate subspace offset
        code.writeline("input_offset = 0")
        for i in range(indices_len):
            comp = [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
            code.writeline(
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)})"
            )
            code.writeline(
                f"cur_index{i} = tl.where(cur_index{i} < 0, "
                f"cur_index{i} + input_shape{i}, cur_index{i})"
            )
            code.writeline(
                f"input_offset = input_offset + cur_index{i} * input_stride{i}"
            )
        # 1D contiguous post segment
        code.writeline("offsets = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)")
        code.writeline("mask = offsets < N")
        code.writeline(
            "v = tl.load(input_ptr + input_offset + offsets, mask=mask, other=0.0)"
        )
        code.writeline("tl.store(out_ptr + pid0 * N + offsets, v, mask=mask)")
    code.newline()
    code.newline()
    return code


def _generate_segment_wrapper(
    inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
):
    code.writeline(f"def {wrapper_name}(input, indices, out):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        code.writeline("grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_SIZE1']))")
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
            args += ["M,", "N,"]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_unsafe_index_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    # Multi-index (indices_len < inp_rank) -> 1D contiguous-segment kernel
    # (avoids the 2D-broadcast pathology on large post dims).
    if indices_len < inp_rank:
        return _generate_segment_kernel(
            inp_rank, indices_len, index_rank, kernel_name, code
        )
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
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [f"out_stride{i}," for i in range(index_rank + inp_rank - indices_len)]
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
        code.writeline("pid1 = ext.program_id(axis=1)")
        code.writeline(
            "offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]"
        )
        if inp_rank == indices_len:
            code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
        else:
            code.writeline(
                "offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]"
            )
        code.newline()
        code.writeline("cur_idx = offset0")
        for i in range(index_rank - 1, -1, -1):
            code.writeline(f"indices_idx{i} = cur_idx % indices0_shape{i}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{i}")
        code.newline()
        code.writeline("cur_idx = offset1")
        for i in range(inp_rank - 1, indices_len - 1, -1):
            code.writeline(f"input_idx{i} = cur_idx % input_shape{i}")
            code.writeline(f"cur_idx = cur_idx // input_shape{i}")
        code.newline()
        code.writeline("mask0 = offset0 < M")
        for i in range(indices_len):
            comp = [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
            code.writeline(
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0)"
            )
            # Wrap negative indices in-kernel (avoids host torch.where overhead).
            code.writeline(
                f"cur_index{i} = tl.where(cur_index{i} < 0, "
                f"cur_index{i} + input_shape{i}, cur_index{i})"
            )
        code.newline()
        # A bounds mask is kept for NPU robustness: without it, masked
        # (out-of-range) lanes carry garbage input offsets and the NPU faults
        # (zero-tolerance OOB). For valid indices it is all-True, so it is cheap.
        index_mask = [
            f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
            for i in range(indices_len)
        ]
        code.writeline(f"index_mask = {' & '.join(index_mask)}")
        code.writeline("mask1 = offset1 < N")
        code.writeline("mask = index_mask & mask0 & mask1")
        code.newline()
        comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
        comp += [
            f"input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)
        ]
        code.writeline(f"input_offset = {' + '.join(comp)}")
        comp = [f"indices_idx{i} * out_stride{i}" for i in range(index_rank)]
        comp += [
            f"input_idx{indices_len + i} * out_stride{index_rank + i}"
            for i in range(inp_rank - indices_len)
        ]
        code.writeline(f"out_offset = {' + '.join(comp)}")
        # Clamp out-of-range lanes to a valid offset: without a bounds-check
        # mask (unsafe), masked lanes would otherwise carry garbage addresses
        # and the NPU faults on them (zero-tolerance OOB).
        code.writeline("input_offset = tl.where(mask, input_offset, 0)")
        code.writeline("out_offset = tl.where(mask, out_offset, 0)")
        code.newline()
        code.writeline("cur_value = tl.load(input_ptr + input_offset , mask = mask)")
        code.writeline("tl.store(out_ptr + out_offset, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_index_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    if indices_len < inp_rank:
        return _generate_segment_wrapper(
            inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
        )
    code.writeline(f"def {wrapper_name}(input, indices, out):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("out_shape = out.shape")
        code.writeline("out_stride = out.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        code.writeline("BLOCK_SIZE0 = 1024 if M >= 4096 else (256 if M >= 64 else 64)")
        code.writeline("grid = (triton.cdiv(M, BLOCK_SIZE0), triton.cdiv(N, 1024))")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["out,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"out_stride[{i}]," for i in range(index_rank + inp_rank - indices_len)
            ]
            args += [
                "M,",
                "N,",
                "BLOCK_SIZE0=BLOCK_SIZE0,",
                "BLOCK_SIZE1=1024,",
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
    inp_rank = inputs[0].ndim
    tensor_indices = [idx for idx in inputs[1] if idx is not None]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = generate_imports(code)
    generate_unsafe_index_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_index_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class UnsafeIndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, out = args
        full_args = (inp, tensor_indices)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
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

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices = args[0], args[1]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        if indices_len == 0:
            index_rank = 0
        else:
            index_rank = tensor_indices[0].ndim
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_unsafe_index_func = UnsafeIndexFunction()


def _unsafe_index(inp, indices):
    """Code-generated ``_unsafe_index`` matching ``aten._unsafe_index``."""
    logger.debug("GEMS UNSAFE_INDEX")
    original_indices = list(indices)
    indices = list(indices)

    if not indices:
        raise ValueError("at least one index must be provided")

    # Single pass: validate dtype (_unsafe_index rejects bool/int8 masks, unlike
    # ``index`` which converts them) and move cross-device indices.  Done in one
    # loop to cut per-call host overhead -- on Hopper these ops are otherwise
    # launch-overhead bound and the host dispatch was a measurable fraction of
    # the total latency on small shapes.
    processed_indices = []
    for index in indices:
        if index is None:
            processed_indices.append(None)
            continue
        dt = index.dtype
        if dt in (torch.int8, torch.bool):
            raise IndexError("_unsafe_index does not support bool or int8 masks")
        if dt not in (torch.long, torch.int, torch.int32, torch.int64):
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )
        if index.device != inp.device:
            index = index.to(inp.device)
        processed_indices.append(index)
    indices = processed_indices

    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    # Fast path: every index is a tensor (no None).  The indexed dims are then a
    # leading contiguous block, the trailing dims are implicit-None, the subspace
    # is already contiguous and no transpose / post-process permute is needed --
    # so we can skip the None-padding, subspace check and shape-splitting loops
    # below.  This is by far the common case and, on Hopper, the host dispatch
    # those loops add is a large fraction of total latency on small shapes.
    if all(idx is not None for idx in indices):
        n = len(indices)
        tensor_indices = (
            list(torch.broadcast_tensors(*indices)) if n > 1 else list(indices)
        )
        out_shape = list(tensor_indices[0].shape) + list(inp.shape[n:])
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        if out.numel() != 0:
            _unsafe_index_func(inp, tensor_indices, out)
        return out

    has_any_tensor = any(idx is not None for idx in indices)
    starts_with_none = indices[0] is None if indices else False

    # Broadcast tensor indices together.
    tensor_indices = [idx for idx in indices if idx is not None]
    if tensor_indices:
        if len(tensor_indices) > 1:
            tensor_indices = list(torch.broadcast_tensors(*tensor_indices))
        tensor_idx = 0
        for i in range(len(indices)):
            if indices[i] is not None:
                indices[i] = tensor_indices[tensor_idx]
                tensor_idx += 1

    # Pad missing trailing dims with None.
    while len(indices) < inp.ndim:
        indices.append(None)

    # Contiguous-subspace check; transpose if needed (tensor dims first).
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    need_post_process = False
    first_tensor_dim = None
    if not has_contiguous_subspace or (starts_with_none and has_any_tensor):
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        inp = inp.permute(dims).contiguous()
        indices = transposed_indices

        if starts_with_none and has_any_tensor and has_contiguous_subspace:
            need_post_process = True
            for i, idx in enumerate(original_indices):
                if idx is not None:
                    first_tensor_dim = i
                    break

    # Output shape: before + replacement + after.
    before_shape = []
    after_shape = []
    replacement_shape = []
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(inp.shape[dim])
            else:
                before_shape.append(inp.shape[dim])
        else:
            if not replacement_shape:
                replacement_shape = list(index.shape)

    out_shape = before_shape + replacement_shape + after_shape
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # ``tensor_indices`` (broadcast above) already holds exactly the non-None
    # tensors in the order the kernel consumes them: the contiguous-subspace
    # transpose above is a *stable* partition (tensors keep their relative
    # order), so recomputing it from ``indices`` would yield the same list.
    if not tensor_indices:
        return inp.view(*out_shape).contiguous()

    # The generated kernel is selected at code-gen time by the
    # (inp_rank, indices_len, index_rank) key:
    #   * indices_len == inp_rank (all-tensor) -> 2D gather kernel.
    #   * indices_len <  inp_rank (multi-index) -> 1D contiguous-segment kernel
    #     (the 2D-broadcast form degrades to per-element stores on a large post
    #     dimension and is much slower).
    # Both kernels wrap negative indices in-kernel (tl.where), so no host
    # neg-wrap is needed.
    # Skip the kernel launch for an empty output: an empty index would give a
    # grid of (0, ...) programs and fault the device; the post-process permute
    # below still runs to produce the correct output shape.
    if out.numel() != 0:
        _unsafe_index_func(inp, tensor_indices, out)

    if need_post_process:
        index_rank = tensor_indices[0].ndim
        pre_dims = list(range(index_rank, index_rank + first_tensor_dim))
        broadcast_dims = list(range(index_rank))
        post_dims = list(range(index_rank + first_tensor_dim, out.ndim))
        new_order = pre_dims + broadcast_dims + post_dims
        return out.permute(new_order).contiguous()

    # ``out`` was just allocated contiguous and the kernel wrote it in order; no
    # copy is needed (the previous unconditional ``.contiguous()`` only added a
    # redundant dispatch on the common path).
    return out
