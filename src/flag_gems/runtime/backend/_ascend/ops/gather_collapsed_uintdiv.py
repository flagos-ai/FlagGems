import importlib.util
import os
import sys
from typing import List, Tuple

import torch

from flag_gems.utils.code_utils import IndentedBuffer

WARP_LIST = [8, 16, 32, 64]
MEM_LIST = [120 * 1024, 216 * 1024]
BLOCK_SIZE_LIST = [32, 64, 128, 256, 512, 1024, 2048]


CACHE_DIR = os.path.join(os.getcwd(), "__triton_cache__")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
sys.path.append(CACHE_DIR)


def normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} out of range for ndim={ndim}")
    return dim


def apply_prefix_narrows(
    inp: torch.Tensor, narrows: List[Tuple[int, int]]
) -> torch.Tensor:
    for axis, new_size in narrows:
        if new_size == inp.shape[axis]:
            continue
        inp = inp.narrow(axis, 0, new_size)
    return inp


def can_collapse_axes(
    inp: torch.Tensor, index: torch.Tensor, dim: int
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Determine whether we can use the collapsed (3D) gather kernel.
    Gather definition (dim = d):
    Y[t0..tN-1] =
        inp[t0..t_{d-1}, index[t0..tN-1], t_{d+1}..t_{N-1}]

    Shape constraints:
    - For i != d: index.shape[i] <= inp.shape[i]
    - Output only accesses inp at coordinates 0 <= t_i < index.shape[i]

    Collapsed kernel assumption:
    We fold tensor into (Outer, Dim, Inner):
        Outer = ∏_{i<d} shape[i]
        Inner = ∏_{i>d} shape[i]
    The same (off_outer, off_inner) must map consistently
    in inp and index/out (linear isomorphism).

    Policy:
    - For i < dim (outer side):
        allow index.shape[i] <= inp.shape[i].
        If strictly smaller, we can prefix-narrow inp so that
        outer dimensions match and linear mapping remains valid.
    - For i > dim (inner side):
        require exact equality to preserve inner linear mapping.
    """
    if inp.ndim != index.ndim:
        return False, []

    dim = normalize_dim(dim, inp.ndim)
    narrows: List[Tuple[int, int]] = []

    for i in range(inp.ndim):
        if i == dim:
            continue

        inp_i = int(inp.shape[i])
        idx_i = int(index.shape[i])

        if i < dim:
            if idx_i == inp_i:
                continue
            if idx_i < inp_i:
                narrows.append((i, idx_i))
                continue
            return False, []
        else:
            if idx_i != inp_i:
                return False, []

    return True, narrows


LIBDIVIDE_ADD_MARKER = 0x40


def _clz32(x: int) -> int:
    return 32 - x.bit_length() if x else 32


def calc_magic_u32_libdivide(d: int):
    """return (magic:uint32, more:uint8)"""
    assert 1 <= d <= 0xFFFFFFFF
    if (d & (d - 1)) == 0:
        shift = d.bit_length() - 1
        return 0, shift & 0xFF
    floor_log_2_d = 31 - _clz32(d)
    two_to = 1 << (32 + floor_log_2_d)
    proposed_m = two_to // d
    rem = two_to - proposed_m * d
    e = d - rem
    two_power = 1 << floor_log_2_d
    if e < two_power:
        magic = (proposed_m + 1) & 0xFFFFFFFF
        more = floor_log_2_d & 0xFF
        return magic, more
    else:
        proposed_m2 = proposed_m * 2
        twice_rem = rem * 2
        if twice_rem >= d or twice_rem < rem:
            proposed_m2 += 1
        magic = (proposed_m2 + 1) & 0xFFFFFFFF
        more = (floor_log_2_d | LIBDIVIDE_ADD_MARKER) & 0xFF
        return magic, more


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    return code


def generate_collapsed_device_functions(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("# Device Functions for Fast Division (collapsed path)")
    code.newline()

    code.writeline("@triton.jit")
    code.writeline("def fast_divide_shift(n, shift):")
    with code.indent():
        code.writeline("return n >> shift")
    code.newline()

    code.writeline("@triton.jit")
    code.writeline("def fast_divide_mul_noadd(n, magic, shift):")
    with code.indent():
        code.writeline("return (tl.umulhi(n, magic) >> shift)")
    code.newline()

    code.writeline("@triton.jit")
    code.writeline("def fast_divide_mul_add(n, magic, shift):")
    with code.indent():
        code.writeline("q0 = tl.umulhi(n, magic)")
        code.writeline("t = ((n - q0) >> 1) + q0")
        code.writeline("return (t >> shift)")
    code.newline()

    return code


def _collapsed_3d_views(
    inp: torch.Tensor, dim: int, index: torch.Tensor, out: torch.Tensor
):
    dim = normalize_dim(dim, inp.ndim)

    # Collapse Axes to 3D: (Outer, Dim, Inner)
    idx_outer = 1
    for i in range(dim):
        idx_outer *= index.shape[i]
    idx_inner = 1
    for i in range(dim + 1, index.ndim):
        idx_inner *= index.shape[i]

    inp_outer = 1
    for i in range(dim):
        inp_outer *= inp.shape[i]
    inp_inner = 1
    for i in range(dim + 1, inp.ndim):
        inp_inner *= inp.shape[i]

    inp_3d = inp.contiguous().view(inp_outer, inp.shape[dim], inp_inner)
    idx_3d = index.contiguous().view(idx_outer, index.shape[dim], idx_inner)
    out_3d = out.view(idx_outer, index.shape[dim], idx_inner)

    SIZE_OUTER = idx_outer
    SIZE_DIM = idx_3d.shape[1]
    SIZE_INNER = idx_inner

    return inp_3d, idx_3d, out_3d, SIZE_OUTER, SIZE_DIM, SIZE_INNER


def generate_collapsed_kernel(
    kernel_name: str, div_kinds: List[str], code: IndentedBuffer
) -> IndentedBuffer:
    code.newline()

    code.writeline("# Autotune Configuration Lists")
    code.writeline("WARP_LIST = [8, 16, 32]")
    code.writeline("MEM_LIST = [120 * 1024, 216 * 1024]")
    code.writeline("BLOCK_SIZE_LIST = [32, 64, 128, 256, 512, 1024, 2048]")
    code.writeline("REORDER_LIST = [True, False]")
    code.newline()

    code.writeline("@triton.autotune(configs=[")
    with code.indent():
        code.writeline(
            "triton.Config("
            "kwargs={'BLOCK_SIZE': size, 'shared_mem_dynamic_size': localmem, "
            "'enable_simt_reorder_instruction': is_reorder}, num_warps=warp)"
        )
        code.writeline("for warp in WARP_LIST")
        code.writeline("for localmem in MEM_LIST")
        code.writeline("for size in BLOCK_SIZE_LIST")
        code.writeline("for is_reorder in REORDER_LIST")
    code.writeline("],")
    code.writeline("key=['num_elements'], ")
    code.writeline("warmup=25, ")
    code.writeline("rep=100)")

    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = [
            "inp_ptr, ",
            "index_ptr, ",
            "out_ptr, ",
            "SIZE_OUTER, ",
            "SIZE_DIM, ",
            "SIZE_INNER, ",
            "stride_inp_outer, ",
            "stride_inp_dim, ",
            "stride_inp_inner, ",
            "stride_idx_outer, ",
            "stride_idx_dim, ",
            "stride_idx_inner, ",
            "stride_out_outer, ",
            "stride_out_dim, ",
            "stride_out_inner, ",
            "inner_magic: tl.uint32, ",
            "inner_shift: tl.uint32, ",
            "dim_magic: tl.uint32, ",
            "dim_shift: tl.uint32, ",
            "num_elements, ",
            "with_negative_index: tl.constexpr, ",
            "BLOCK_SIZE: tl.constexpr, ",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("num_programs = tl.num_programs(0)")
        code.writeline("elements_per_prog = tl.cdiv(num_elements, num_programs)")
        code.writeline("prog_start = pid * elements_per_prog")
        code.writeline(
            "prog_end = tl.minimum(prog_start + elements_per_prog, num_elements)"
        )
        code.newline()

        code.writeline("for block_start in range(prog_start, prog_end, BLOCK_SIZE):")
        with code.indent():
            code.writeline("offsets = block_start + tl.arange(0, BLOCK_SIZE)")
            code.writeline("mask = offsets < prog_end")
            code.newline()
            code.writeline("idx_val = tl.load(index_ptr + offsets, mask=mask, other=0)")

            code.newline()
            code.writeline("if with_negative_index:")
            with code.indent():
                code.writeline(
                    "idx_val = tl.where(idx_val < 0, idx_val + SIZE_DIM, idx_val)"
                )
            code.newline()

            # offsets -> (off_outer, off_dim, off_inner)
            # q1 = offsets // SIZE_INNER
            # code.writeline("q1 = offsets // SIZE_INNER")
            if div_kinds[0] == "S":
                code.writeline("q1 = fast_divide_shift(offsets, inner_shift)")
            elif div_kinds[0] == "A":
                code.writeline(
                    "q1 = fast_divide_mul_add(offsets, inner_magic, inner_shift)"
                )
            else:
                code.writeline(
                    "q1 = fast_divide_mul_noadd(offsets, inner_magic, inner_shift)"
                )

            code.writeline("off_inner = offsets - q1 * SIZE_INNER")
            code.writeline("tmp = q1")
            code.newline()

            # q2 = tmp // SIZE_DIM
            # code.writeline("q2 = tmp // SIZE_DIM")
            if div_kinds[1] == "S":
                code.writeline("q2 = fast_divide_shift(tmp, dim_shift)")
            elif div_kinds[1] == "A":
                code.writeline("q2 = fast_divide_mul_add(tmp, dim_magic, dim_shift)")
            else:
                code.writeline("q2 = fast_divide_mul_noadd(tmp, dim_magic, dim_shift)")

            code.writeline("off_dim = tmp - q2 * SIZE_DIM")
            code.writeline("off_outer = q2")
            code.newline()

            code.writeline("inp_off = (")
            with code.indent():
                code.writeline("off_outer * stride_inp_outer")
                code.writeline("+ idx_val * stride_inp_dim")
                code.writeline("+ off_inner * stride_inp_inner")
            code.writeline(")")
            code.writeline("val = tl.load(inp_ptr + inp_off, mask=mask, other=0.0)")
            code.newline()

            code.writeline("tl.store(out_ptr + offsets, val, mask=mask)")

    code.newline()
    return code


def generate_collapsed_wrapper(
    wrapper_name: str, kernel_name: str, code: IndentedBuffer
) -> IndentedBuffer:
    code.writeline(
        f"def {wrapper_name}("
        f"inp, index, out, grid, inner_magic, inner_shift, dim_magic, dim_shift, with_negative_index):"
    )
    with code.indent():
        code.writeline("inp_shape = inp.shape")
        code.writeline("inp_stride = inp.stride()")
        code.writeline("index_shape = index.shape")
        code.writeline("index_stride = index.stride()")
        code.writeline("out_stride = out.stride()")
        code.writeline("num_elements = out.numel()")
        code.newline()

        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = [
                "inp, ",
                "index, ",
                "out, ",
                "index_shape[0], ",  # SIZE_OUTER
                "index_shape[1], ",  # SIZE_DIM
                "index_shape[2], ",  # SIZE_INNER
                "inp_stride[0], ",
                "inp_stride[1], ",
                "inp_stride[2], ",
                "index_stride[0], ",
                "index_stride[1], ",
                "index_stride[2], ",
                "out_stride[0], ",
                "out_stride[1], ",
                "out_stride[2], ",
                "inner_magic, ",
                "inner_shift, ",
                "dim_magic, ",
                "dim_shift, ",
                "num_elements, ",
                "with_negative_index, ",
                "force_simt_only=False, ",
            ]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return out")
    code.newline()
    return code


def generate_collapsed_code(
    wrapper_name: str, kernel_name: str, div_kinds: List[str]
) -> str:
    code = IndentedBuffer()
    code = generate_imports(code)
    code = generate_collapsed_device_functions(code)
    code = generate_collapsed_kernel(kernel_name, div_kinds, code)
    code = generate_collapsed_wrapper(wrapper_name, kernel_name, code)
    return code.getvalue()


class CollapsedGatherFunction:
    def __init__(self):
        self.overloads = {}

    def __call__(
        self, inp, index, out, grid, magic_shift_map=None, with_negative_index=False
    ):
        assert inp.ndim == 3
        assert index.ndim == 3
        assert out.ndim == 3

        if magic_shift_map is None:
            # two divisors only: SIZE_INNER, SIZE_DIM
            inner_magic, inner_more = calc_magic_u32_libdivide(int(index.shape[2]))
            dim_magic, dim_more = calc_magic_u32_libdivide(int(index.shape[1]))
        else:
            (inner_magic, inner_more), (dim_magic, dim_more) = magic_shift_map

        inner_shift = int(inner_more) & 0x1F
        dim_shift = int(dim_more) & 0x1F

        inner_kind = (
            "S" if int(inner_magic) == 0 else ("A" if (int(inner_more) & 0x40) else "M")
        )
        dim_kind = (
            "S" if int(dim_magic) == 0 else ("A" if (int(dim_more) & 0x40) else "M")
        )

        pattern = inner_kind + dim_kind
        key = f"collapsed_pat_{pattern}"

        if key not in self.overloads:
            kernel_name = f"_gather_collapsed_kernel_{pattern}"
            wrapper_name = f"_gather_collapsed_wrapper_{pattern}"

            src_code = generate_collapsed_code(
                wrapper_name, kernel_name, [inner_kind, dim_kind]
            )

            file_name = f"{key}.py"
            file_path = os.path.join(CACHE_DIR, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(src_code)

            spec = importlib.util.spec_from_file_location(
                f"dynamic_collapsed_mod_{key}", file_path
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)

            self.overloads[key] = getattr(mod, wrapper_name)

        return self.overloads[key](
            inp,
            index,
            out,
            grid,
            inner_magic,
            inner_shift,
            dim_magic,
            dim_shift,
            with_negative_index,
        )


collapsed_gather = CollapsedGatherFunction()


def gather_collapsed(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    out: torch.Tensor,
    grid_fn,
    return_run_kernel: bool = True,
    with_negative_index: bool = False,
):
    if out.shape != index.shape:
        raise ValueError(f"out.shape {out.shape} must equal index.shape {index.shape}")

    dim = normalize_dim(dim, inp.ndim)
    inp_3d, idx_3d, out_3d, SIZE_OUTER, SIZE_DIM, SIZE_INNER = _collapsed_3d_views(
        inp, dim, index, out
    )

    def _run_kernel():
        collapsed_gather(
            inp_3d, idx_3d, out_3d, grid_fn, with_negative_index=with_negative_index
        )

    if return_run_kernel:
        return _run_kernel

    _run_kernel()
