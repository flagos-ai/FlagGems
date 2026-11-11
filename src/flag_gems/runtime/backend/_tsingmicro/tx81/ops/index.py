import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

_INT_INDEX_DTYPES = (torch.int32, torch.int64, torch.long, torch.int)
# inp larger than this uses device-native aten.index; smaller tensors use CPU fallback.
_INDEX_CPU_FALLBACK_MAX_BYTES = int(
    os.environ.get("FLAGGEMS_INDEX_CPU_FALLBACK_MAX_BYTES", str(1 << 30))
)


def _inp_nbytes(inp: torch.Tensor) -> int:
    return inp.numel() * inp.element_size()


def _native_index(inp: torch.Tensor, indices: list) -> torch.Tensor:
    """Device-native aten.index (avoid CPU round-trip on large tensors)."""
    return torch.ops.aten.index(inp, indices)


def _tail_dense_from(inp: torch.Tensor, start_dim: int) -> bool:
    """True when inp[..., *] from ``start_dim`` is a packed contiguous tail."""
    if start_dim >= inp.ndim:
        return True
    stride = inp.stride()
    shape = inp.shape
    step = 1
    for d in range(inp.ndim - 1, start_dim - 1, -1):
        if stride[d] != step:
            return False
        step *= shape[d]
    return True


def _try_fast_leading_two(inp: torch.Tensor, indices: list):
    """Fast path for ``inp[i0, i1]`` (QSA paged gather: cache[phys, page_offset]).

    Folds the first two axes into a linear dim0 gather and reuses
    ``index_select``'s 16-CTA kernel. Supports fat page stride
    (``stride0 > shape1 * stride1``), e.g. field ``[40960,512,1,256]``
    with ``stride0=262144``.
    """
    if len(indices) != 2:
        return None
    i0, i1 = indices[0], indices[1]
    if i0 is None or i1 is None:
        return None
    if i0.dtype not in _INT_INDEX_DTYPES or i1.dtype not in _INT_INDEX_DTYPES:
        return None
    if inp.ndim < 2:
        return None
    if not _tail_dense_from(inp, start_dim=2):
        return None

    s0 = int(inp.stride(0))
    s1 = int(inp.stride(1))
    if s1 <= 0 or s0 % s1 != 0:
        return None
    page_span = s0 // s1
    if page_span < int(inp.shape[1]):
        return None

    try:
        i0b, i1b = torch.broadcast_tensors(i0, i1)
    except RuntimeError:
        return None

    flat_idx = (
        i0b.to(dtype=torch.int64) * page_span + i1b.to(dtype=torch.int64)
    ).reshape(-1)
    flat_idx = flat_idx.contiguous()

    tail_shape = tuple(inp.shape[2:])
    n_rows = (int(inp.shape[0]) - 1) * page_span + int(inp.shape[1])
    if n_rows <= 0:
        return None

    from .index_select import dim0_rows_are_dense, index_select

    if tail_shape:
        row_view = inp.as_strided(
            size=(n_rows, *tail_shape),
            stride=(s1, *inp.stride()[2:]),
        )
        if not dim0_rows_are_dense(row_view):
            return None
        out = index_select(row_view, 0, flat_idx)
        return out.view(*i0b.shape, *tail_shape)

    # 2D inp[i0, i1] → scalar cells (e.g. block_table[req, page]).
    # index_select dim0 on a dense [n_rows, 1] view.
    row_view2 = inp.as_strided(size=(n_rows, 1), stride=(s1, 1))
    if not dim0_rows_are_dense(row_view2):
        return None
    out = index_select(row_view2, 0, flat_idx).reshape(-1)
    return out.view(*i0b.shape)


def _try_fast_dim0(inp, indices):
    """Reuse index_select 16-CTA gather for common advanced-index patterns."""
    fast2 = _try_fast_leading_two(inp, indices)
    if fast2 is not None:
        return fast2

    if len(indices) != 1:
        return None
    index = indices[0]
    if index is None:
        return None
    if index.dtype not in _INT_INDEX_DTYPES:
        return None
    from .index_select import dim0_rows_are_dense, index_select

    if not dim0_rows_are_dense(inp):
        return None

    # Multi-dim index (e.g. [B,T] gather on embedding [V,H]): flatten → select → view.
    # Profiler hotspot: inp[1048576,64] bf16 with 2D indices fell to CPU (~52ms) when
    # we rejected ndim>1.
    index_shape = tuple(index.shape)
    flat = index.reshape(-1) if index.ndim != 1 else index
    if not flat.is_contiguous():
        flat = flat.contiguous()
    out = index_select(inp, 0, flat)
    if index.ndim == 1:
        return out
    return out.view(*index_shape, *inp.shape[1:])


def _try_fast_2d_dim0(inp, indices):
    """Backward-compatible alias."""
    return _try_fast_dim0(inp, indices)


def _fallback_cpu(inp, indices):
    """Mirror torch_txda PrivateUse1 index backend fallback (same idea as index_select)."""
    device = inp.device
    indices_cpu = [None if idx is None else idx.cpu() for idx in indices]
    out = torch.ops.aten.index(inp.cpu(), indices_cpu)
    return out.to(device=device, non_blocking=False)


def get_max_rank_shape(indices: List[torch.Tensor]) -> List[int]:
    # Filter out None values (basic indexing markers)
    tensor_indices = [idx for idx in indices if idx is not None]
    if len(tensor_indices) == 0:
        return []
    max_rank = max([len(index.shape) for index in tensor_indices])
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in tensor_indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0:
                max_num = max(max_num, index.shape[axis])  #
        shape[max_rank - 1 - i] = max_num
    return shape


def broadcast_indices(indices, target_shape):
    for i, index in enumerate(indices):
        if index is not None and tuple(index.shape) != tuple(target_shape):
            indices[i] = torch.broadcast_to(index, target_shape)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")

    code.newline()
    code.newline()
    return code


def generate_index_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline("@libtuner(")
    with code.indent():
        code.writeline('configs=runtime.get_tuned_config("index"),')
        code.writeline('key=["M", "N"],')
        code.writeline('strategy=["align32", "align32"],')
        code.writeline("warmup=1,")
        code.writeline("rep=2,")
    code.writeline(")")
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
        code.writeline("pid0 = tle.program_id(axis=0)")
        code.writeline("pid1 = tle.program_id(axis=1)")
        code.writeline(
            "offset0 = pid0.to(tl.int64)* BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0).to(tl.int64)[:, None]"
        )
        if inp_rank == indices_len:
            code.writeline(
                "offset1 = pid1.to(tl.int64) * 1 + tl.arange(0, 1).to(tl.int64)[None, :]"
            )
        else:
            code.writeline(
                "offset1 = pid1.to(tl.int64) * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1).to(tl.int64)[None, :]"
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
        code.newline()
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
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(M, meta['BLOCK_SIZE0']), ")
            code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']), ")
        code.writeline(")")
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
            args += ["M,", "N,"]
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
    # Filter out None values to get actual tensor indices
    tensor_indices = [idx for idx in inputs[1] if idx is not None]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = generate_imports(code)
    generate_index_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_index_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexFunction:
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
                "_index_wrapper",
                "_index_jit_function",
                code,
            )

            file_name = f"index_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_wrapper")
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


_index_func = IndexFunction()


def index(inp, indices):
    logger.debug("GEMS_TSINGMICRO INDEX")
    indices = list(indices)

    if not indices:
        raise ValueError("at least one index must be provided")

    fast = _try_fast_dim0(inp, indices)
    if fast is not None:
        return fast
    # if _inp_nbytes(inp) > _INDEX_CPU_FALLBACK_MAX_BYTES:
    #     return _native_index(inp, indices)
    return _fallback_cpu(inp, indices)
