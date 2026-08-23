import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.triton_version_utils import _triton_version_at_least

logger = logging.getLogger(__name__)

_TRITON_SUPPORTS_BF16_ATOMIC_ADD = _triton_version_at_least(3, 4)
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code


def generate_index_add_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("src,")
            code.writeline("out,")
            code.writeline("N,")
            code.writeline("inp_numel,")
            code.writeline("inp_stride_dim,")
            code.writeline("inp_shape_dim,")
            code.writeline("src_shape_dim,")
            code.writeline("delta,")
            code.writeline("alpha,")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"src_shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for src")

            code.writeline("BLOCK_SIZE: tl.constexpr,")

        code.writeline("):")

        # Kernel Code
        with code.indent():
            code.writeline("pid = tl.program_id(axis=0)")
            code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            code.writeline("mask = offsets < N")

            for i in range(rank - 1, -1, -1):
                code.writeline(f"src_offset{i} = offsets % src_shape_{i}")
                code.writeline(f"offsets = offsets // src_shape_{i}")
            code.newline()
            comp = [f"src_offset{i} * src_stride_{i}" for i in range(rank)]
            code.writeline(f"src_offset = {' + '.join(comp)}")

            code.writeline("pre_cal = (inp_stride_dim * src_shape_dim)")

            # index add
            code.writeline("pre_idx = (src_offset // pre_cal).to(tl.int64)")
            code.writeline(
                "dim_idx = (src_offset % pre_cal // inp_stride_dim).to(tl.int64)"
            )
            code.writeline(
                "src_dim_idx = (tl.load(index + dim_idx, mask=mask, other=0)).to(tl.int64)"
            )
            code.writeline(
                'assert src_dim_idx >= 0 and src_dim_idx < inp_shape_dim, "0 <= index < self.size(dim)"'
            )
            code.writeline(
                "input_idx = (src_offset + (delta * pre_idx + src_dim_idx - dim_idx) * inp_stride_dim).to(tl.int64)"
            )

            code.writeline("input_mask = input_idx < inp_numel")
            code.writeline(
                "add_on = tl.load(src + src_offset, mask=mask, other=0) * alpha"
            )
            code.writeline(
                "tl.atomic_add(out + input_idx, add_on, mask=input_mask, sem='relaxed')"
            )
            # Older Triton versions receive FP32 tensors from the wrapper.

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    # out, index, src, dim, inp_stride_dim, src_shape_dim, delta, N, inp.numel(), alpha
    parameters: List[str] = []
    parameters.append("out")
    parameters.append("index")
    parameters.append("src")
    parameters.append("dim")
    parameters.append("inp_stride_dim")
    parameters.append("inp_shape_dim")
    parameters.append("src_shape_dim")
    parameters.append("delta")
    parameters.append("N")
    parameters.append("inp_numel")
    parameters.append("alpha")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name} ({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("src_strides = list(src.stride())")
        code.writeline("src_shapes = list(src.shape)")

        # kernel launch
        code.writeline("BLOCK_SIZE = 128")  # BLOCK_SIZE setting
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline(
                "index, src, out, N, inp_numel, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, alpha, "
            )
            if rank > 0:
                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [out, index, src, dim, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, N, inp.numel(), alpha]
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_index_add_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexAddFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_index_add_wrapper",
                "_index_add_jit_function",
                code,
            )

            file_name = f"index_add_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_add_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_index_add_func = IndexAddFunction()


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _assert_index_in_bounds(index, upper_bound):
    # Validate before scatter so an in-place call leaves its input unchanged on error.
    lower, upper = torch.ops.aten.aminmax.default.redispatch(
        _FALLBACK_KEYSET, index, dim=None, keepdim=False
    )
    assert (
        lower.item() >= 0 and upper.item() < upper_bound
    ), "0 <= index < self.size(dim)"


def _can_use_contiguous_suffix_path(inp, dim, index, src):
    if src.numel() == 0:
        return False
    if not (
        inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False

    suffix_size = _volume(src.shape[dim + 1 :])
    return suffix_size > 1


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_flat"),
    key=["total_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_flat_kernel(
    out,
    index,
    src,
    total_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    offsets = ext.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_count

    cols = offsets % suffix_size
    rows = offsets // suffix_size
    src_dim_idx = rows % index_len
    prefix_idx = rows // index_len
    dst_dim_idx = tl.load(index + src_dim_idx, mask=mask, other=0).to(tl.int64)
    valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(out + out_offsets, values * alpha, mask=valid, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_fp16_flat"),
    key=["total_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_fp16_flat_kernel(
    out,
    index,
    src,
    total_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_count

    cols = offsets % suffix_size
    rows = offsets // suffix_size
    src_dim_idx = rows % index_len
    prefix_idx = rows // index_len
    dst_dim_idx = tl.load(index + src_dim_idx, mask=mask, other=0).to(tl.int64)
    valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    tl.atomic_add(out + out_offsets, values * alpha, mask=valid, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_tile"),
    key=["row_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_tile_kernel(
    out,
    index,
    src,
    row_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < row_count
    col_mask = cols < suffix_size
    mask = row_mask & col_mask

    src_dim_idx = rows % index_len
    prefix_idx = rows // index_len
    dst_dim_idx = tl.load(index + src_dim_idx, mask=row_mask, other=0).to(tl.int64)
    valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(out + out_offsets, values * alpha, mask=valid, sem="relaxed")


def _run_contiguous_suffix_flat_path(
    out, dim, index, src, alpha, use_fp16_config=False
):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    total_count = row_count * suffix_size
    grid = lambda meta: (triton.cdiv(total_count, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        if use_fp16_config:
            _index_add_contiguous_suffix_fp16_flat_kernel[grid](
                out,
                index,
                src,
                total_count,
                index.numel(),
                out.size(dim),
                suffix_size,
                alpha,
            )
        else:
            _index_add_contiguous_suffix_flat_kernel[grid](
                out,
                index,
                src,
                total_count,
                index.numel(),
                out.size(dim),
                suffix_size,
                alpha,
                ACCUMULATE_FP32=(
                    out.dtype == torch.float32 and src.dtype == torch.bfloat16
                ),
            )
    return out


def _run_contiguous_suffix_tile_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    grid = lambda meta: (
        triton.cdiv(row_count, meta["BLOCK_M"]),
        triton.cdiv(suffix_size, meta["BLOCK_N"]),
    )
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_tile_kernel[grid](
            out,
            index,
            src,
            row_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ACCUMULATE_FP32=(
                out.dtype == torch.float32 and src.dtype == torch.bfloat16
            ),
        )
    return out


def _run_contiguous_suffix_path(out, dim, index, src, alpha):
    # View contiguous tensors as [prefix, index_len, suffix] and scatter-add
    # dense suffix tiles without generic rank/stride address decomposition.
    suffix_size = _volume(src.shape[dim + 1 :])
    if src.dtype == torch.float16 and suffix_size <= 512:
        return _run_contiguous_suffix_flat_path(
            out, dim, index, src, alpha, use_fp16_config=True
        )
    if suffix_size <= 64:
        return _run_contiguous_suffix_flat_path(out, dim, index, src, alpha)
    return _run_contiguous_suffix_tile_path(out, dim, index, src, alpha)


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_HYGON INDEX_ADD")
    normalized_dim = dim % inp.ndim if -inp.ndim <= dim < inp.ndim else dim
    if _can_use_contiguous_suffix_path(inp, normalized_dim, index, src):
        _assert_index_in_bounds(index, inp.size(dim))
        accumulate_fp32 = (
            inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
        )
        out = inp.float() if accumulate_fp32 else inp.clone()
        res = _run_contiguous_suffix_path(
            out, normalized_dim, index.contiguous(), src, alpha
        )
        if res is not None:
            return res.to(inp.dtype) if accumulate_fp32 else res

    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    accumulate_fp32 = (
        inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
    )
    out = inp.float() if accumulate_fp32 else inp.clone()
    src_for_kernel = src.float() if accumulate_fp32 else src

    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src_for_kernel.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src_for_kernel.numel()

    _index_add_func(
        out,
        index,
        src_for_kernel,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
        alpha,
    )
    return out.to(inp.dtype) if accumulate_fp32 else out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_HYGON INDEX_ADD_")
    normalized_dim = dim % inp.ndim if -inp.ndim <= dim < inp.ndim else dim
    if _can_use_contiguous_suffix_path(inp, normalized_dim, index, src):
        _assert_index_in_bounds(index, inp.size(dim))
        accumulate_fp32 = (
            inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
        )
        out = inp.float() if accumulate_fp32 else inp
        res = _run_contiguous_suffix_path(
            out, normalized_dim, index.contiguous(), src, alpha
        )
        if res is not None:
            if accumulate_fp32:
                inp.copy_(res)
            return inp

    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    accumulate_fp32 = (
        inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
    )
    out = inp.float() if accumulate_fp32 else inp
    src_for_kernel = src.float() if accumulate_fp32 else src

    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src_for_kernel.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src_for_kernel.numel()

    _index_add_func(
        out,
        index,
        src_for_kernel,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
        alpha,
    )
    if accumulate_fp32:
        inp.copy_(out)
    return inp
