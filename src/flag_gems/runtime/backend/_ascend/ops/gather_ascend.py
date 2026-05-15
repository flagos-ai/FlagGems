import functools
import importlib.util
import logging
import os
import sys
from typing import List, Tuple

import torch

from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)
LIBDIVIDE_32_SHIFT_MASK = 0x1F
LIBDIVIDE_ADD_MARKER = 0x40

CACHE_DIR = os.path.join(os.getcwd(), "__triton_cache__")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
sys.path.append(CACHE_DIR)


def _clz32(x: int) -> int:
    return 32 - x.bit_length() if x else 32


def calc_magic_u32_libdivide(d: int) -> Tuple[int, int]:
    """
    Compute the libdivide (u32) fast division parameters for a given divisor d.
    Returns: (magic:uint32, more:uint8)

    - magic == 0 indicates the power-of-two path (shift only)
    - the lower 5 bits of `more` represent the shift value
    - bit6 (0x40) of `more` indicates the add_marker flag
    """
    if not (1 <= d <= 0xFFFFFFFF):
        raise ValueError(f"d must be in [1, 2^32-1], got {d}")

    # pow2 -> shift path
    if (d & (d - 1)) == 0:
        shift = d.bit_length() - 1
        return 0, shift & 0xFF

    floor_log_2_d = 31 - _clz32(d)

    # 2^(32+floor_log_2_d)
    two_to = 1 << (32 + floor_log_2_d)

    proposed_m = two_to // d
    rem = two_to - proposed_m * d
    e = d - rem
    two_power = 1 << floor_log_2_d

    if e < two_power:
        # no add marker
        magic = (proposed_m + 1) & 0xFFFFFFFF
        more = floor_log_2_d & 0xFF
        return magic, more
    else:
        # add marker
        proposed_m2 = proposed_m * 2
        twice_rem = rem * 2
        if twice_rem >= d or twice_rem < rem:
            proposed_m2 += 1
        magic = (proposed_m2 + 1) & 0xFFFFFFFF
        more = (floor_log_2_d | LIBDIVIDE_ADD_MARKER) & 0xFF
        return magic, more


@functools.lru_cache(maxsize=128)
def get_all_magics(shape_tuple: Tuple[int, ...]) -> Tuple[List[int], List[int]]:
    magic_list, more_list = [], []
    for d in shape_tuple:
        magic, more = calc_magic_u32_libdivide(int(d))
        magic_list.append(magic)
        more_list.append(more)
    return magic_list, more_list


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    return code


def generate_device_functions(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline(
        "# Device Functions for Fast Division (assume uint32 inputs, no casts/masks)"
    )
    code.newline()

    # shift-only (magic==0)
    code.writeline("@triton.jit")
    code.writeline("def fast_divide_shift(n, shift):")
    with code.indent():
        code.writeline("return n >> shift")
    code.newline()

    # mul-noadd (add_marker==0)
    code.writeline("@triton.jit")
    code.writeline("def fast_divide_mul_noadd(n, magic, shift):")
    with code.indent():
        code.writeline("return (tl.umulhi(n, magic) >> shift)")
    code.newline()

    # mul-add (add_marker==1)
    code.writeline("@triton.jit")
    code.writeline("def fast_divide_mul_add(n, magic, shift):")
    with code.indent():
        code.writeline("q0 = tl.umulhi(n, magic)")
        code.writeline("t = ((n - q0) >> 1) + q0")
        code.writeline("return (t >> shift)")
    code.newline()

    return code


def generate_gather_kernel(
    rank: int, kernel_name: str, div_kinds: List[str], code: IndentedBuffer
) -> IndentedBuffer:
    code.newline()

    # Autotune lists
    code.writeline("# Autotune Configuration Lists")
    code.writeline("WARP_LIST = [8, 16, 32]")
    code.writeline("MEM_LIST = [120 * 1024, 216 * 1024]")
    code.writeline("BLOCK_SIZE_LIST = [32, 64, 128, 256, 512, 1024]")
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
    code.writeline("rep=100) ")

    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = [
            "inp_ptr, ",
            "index_ptr, ",
            "out_ptr, ",
        ]
        # Unroll shapes and strides in signature to avoid metadata loads
        args += [f"inp_shape{i}, " for i in range(rank)]
        args += [f"index_shape{i}, " for i in range(rank)]

        # libdivide params per dimension
        args += [f"index_magic{i}: tl.uint32, " for i in range(rank)]
        args += [f"index_more{i}: tl.uint32, " for i in range(rank)]

        args += [f"inp_stride{i}, " for i in range(rank)]
        args += [f"index_stride{i}, " for i in range(rank)]
        args += [f"out_stride{i}, " for i in range(rank)]

        args += [
            "dim: tl.constexpr, ",
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

        code.writeline(
            "# Block-Stride Loop (Processing contiguous chunks for better cache hit rate)"
        )
        code.writeline("for block_start in range(prog_start, prog_end, BLOCK_SIZE):")
        with code.indent():
            code.writeline("offsets = block_start + tl.arange(0, BLOCK_SIZE)")
            code.writeline("mask = offsets < num_elements")
            code.newline()

            code.writeline(
                "gather_index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int32)"
            )

            code.writeline("base_inp_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int32)")
            code.writeline("cur_offset = offsets.to(tl.int32)")
            code.newline()

            code.writeline("dim_stride = 0")
            code.writeline("dim_size = 0")
            code.newline()

            for i in range(rank - 1, -1, -1):
                if i == 0:
                    # After processing dims [rank-1 .. 1], cur_offset is already in [0, index_shape0).
                    # So: next_offset = cur_offset // index_shape0 == 0, coord_0 == cur_offset.
                    code.writeline("coord_0 = cur_offset")
                    code.writeline("cur_offset = 0")
                else:
                    code.writeline(f"shift = index_more{i}")
                    if div_kinds[i] == "S":
                        code.writeline(
                            "next_offset = fast_divide_shift(cur_offset, shift)"
                        )
                    elif div_kinds[i] == "A":
                        code.writeline(f"magic = index_magic{i}")
                        code.writeline(
                            "next_offset = fast_divide_mul_add(cur_offset, magic, shift)"
                        )
                    else:
                        code.writeline(f"magic = index_magic{i}")
                        code.writeline(
                            "next_offset = fast_divide_mul_noadd(cur_offset, magic, shift)"
                        )

                    code.writeline(
                        f"coord_{i} = cur_offset - next_offset * index_shape{i}"
                    )
                    code.writeline("cur_offset = next_offset")

                code.writeline(f"if dim == {i}:")
                with code.indent():
                    code.writeline(f"dim_stride = inp_stride{i}")
                    code.writeline(f"dim_size = inp_shape{i}")
                code.writeline("else:")
                with code.indent():
                    code.writeline(f"base_inp_offset += coord_{i} * inp_stride{i}")
                code.newline()

            code.writeline("# Handle negative indices")
            code.writeline("if with_negative_index:")
            with code.indent():
                code.writeline(
                    "gather_index = tl.where(gather_index < 0, gather_index + dim_size, gather_index).to(tl.int32)"
                )

            code.writeline(
                "final_inp_offset = base_inp_offset + gather_index * dim_stride"
            )
            code.writeline(
                "val = tl.load(inp_ptr + final_inp_offset, mask=mask, other=0.0)"
            )
            code.writeline("tl.store(out_ptr + offsets, val, mask=mask)")

    code.newline()
    return code


def generate_gather_wrapper(
    rank: int, wrapper_name: str, kernel_name: str, code: IndentedBuffer
) -> IndentedBuffer:
    code.writeline(
        f"def {wrapper_name}(inp, dim, index, out, grid, magic, more, with_negative_index):"
    )
    with code.indent():
        # Extract shapes and strides
        code.writeline("inp_shape = inp.shape")
        code.writeline("inp_stride = inp.stride()")
        code.writeline("index_shape = index.shape")
        code.writeline("index_stride = index.stride()")
        code.writeline("out_stride = out.stride()")
        code.writeline("num_elements = index.numel()")
        code.newline()

        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = [
                "inp, ",
                "index, ",
                "out, ",
            ]
            args += [f"inp_shape[{i}], " for i in range(rank)]
            args += [f"index_shape[{i}], " for i in range(rank)]

            args += [f"magic[{i}], " for i in range(rank)]
            args += [f"more[{i}], " for i in range(rank)]

            args += [f"inp_stride[{i}], " for i in range(rank)]
            args += [f"index_stride[{i}], " for i in range(rank)]
            args += [f"out_stride[{i}], " for i in range(rank)]

            args += [
                "dim, ",
                "num_elements, ",
                "with_negative_index, ",
            ]
            args += [
                "force_simt_only=False, ",
            ]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return out")
    code.newline()
    return code


def generate_code(
    inputs, wrapper_name: str, kernel_name: str, div_kinds: List[str]
) -> str:
    code = IndentedBuffer()
    rank = inputs[0].ndim
    code = generate_imports(code)
    code = generate_device_functions(code)
    code = generate_gather_kernel(rank, kernel_name, div_kinds, code)
    code = generate_gather_wrapper(rank, wrapper_name, kernel_name, code)
    return code.getvalue()


class GatherFunction:
    def __init__(self):
        self.overloads = {}
        self.kernels = {}

    def __call__(
        self, inp, dim, index, out, grid, magic_map=None, with_negative_index=False
    ):
        rank = inp.ndim

        if magic_map is None:
            magic, more = get_all_magics(tuple(index.shape))
        else:
            magic, more = magic_map

        # div_kinds: 'S' (shift-only), 'M' (mul-noadd), 'A' (mul-add)
        div_kinds = []
        for m, mo in zip(magic, more):
            if int(m) == 0:
                div_kinds.append("S")
            elif (int(mo) & 0x40) != 0:
                div_kinds.append("A")
            else:
                div_kinds.append("M")

        pattern = "".join(div_kinds)
        key = f"gather_rank_{rank}_pat_{pattern}"

        if key not in self.overloads:
            kernel_name = f"_gather_kernel_{rank}"
            wrapper_name = f"_gather_wrapper_{rank}"

            src_code = generate_code([inp], wrapper_name, kernel_name, div_kinds)

            file_name = f"{key}.py"
            file_path = os.path.join(CACHE_DIR, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(src_code)

            spec = importlib.util.spec_from_file_location(
                f"dynamic_mod_{key}", file_path
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)

            self.overloads[key] = getattr(mod, wrapper_name)

        return self.overloads[key](
            inp, dim, index, out, grid, magic, more, with_negative_index
        )

    def get_kernel(self, rank: int):
        return self.kernels.get(f"gather_rank_{rank}")


_gather_func = GatherFunction()


def gather(
    inp,
    dim: int,
    index,
    out=None,
    grid_fn=None,
    magic_map=None,
    with_negative_index=False,
):
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    _gather_func(
        inp,
        dim,
        index,
        out,
        grid_fn,
        magic_map=magic_map,
        with_negative_index=with_negative_index,
    )
    return out
