import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.runtime.backend._enflame.gcu400.ops.cat import cat_flat_kernel_4

logger = logging.getLogger(__name__)

KERNEL_NUMEL_THRESHOLD = 524288


@libentry()
@triton.jit(do_not_specialize=["N", "dim_prod_post", "dim_size_out",
                                "dim_offset_a", "dim_offset_b", "dim_offset_c", "dim_offset_d",
                                "total_a", "total_b", "total_c", "total_d"])
def stack_general_kernel_4(
    out_ptr,
    in_ptr_a, in_ptr_b, in_ptr_c, in_ptr_d,
    N,
    dim_prod_post,
    dim_size_out,
    dim_offset_a, dim_offset_b, dim_offset_c, dim_offset_d,
    total_a, total_b, total_c, total_d,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    num_pids = tl.num_programs(0)

    if tid == 0:
        in_ptr = in_ptr_a
        dim_offset = dim_offset_a
        total = total_a
    elif tid == 1:
        in_ptr = in_ptr_b
        dim_offset = dim_offset_b
        total = total_b
    elif tid == 2:
        in_ptr = in_ptr_c
        dim_offset = dim_offset_c
        total = total_c
    else:
        in_ptr = in_ptr_d
        dim_offset = dim_offset_d
        total = total_d

    num_blocks = (total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < total
        pre_idx = off // dim_prod_post
        post_idx = off % dim_prod_post
        out_idx = pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx
        data = tl.load(in_ptr + off, mask=mask)
        tl.store(out_ptr + out_idx, data, mask=mask)


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS GCU400 STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp0 = tensors[0]
    inp0_shape = inp0.shape
    ndim = inp0.dim()

    for i in range(1, len(tensors)):
        t = tensors[i]
        if dim < -ndim - 1 or dim > ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -ndim - 1, ndim, dim
                )
            )
        if t.shape != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {list(inp0_shape)} at entry 0 and {list(t.shape)} at entry {i}"
            )

    if dim < 0:
        dim = dim + ndim + 1

    num_tensors = len(tensors)
    dtype = inp0.dtype
    need_cast = False
    for i in range(1, num_tensors):
        if tensors[i].dtype != dtype:
            dtype = torch.promote_types(dtype, tensors[i].dtype)
            need_cast = True

    out_shape = list(inp0_shape)
    out_shape.insert(dim, num_tensors)
    out = torch.empty(out_shape, dtype=dtype, device=inp0.device)

    numel_per = inp0.numel()

    if need_cast:
        tensors = [t.to(dtype) for t in tensors]

    if dim == 0:
        all_contiguous = all(t.is_contiguous() for t in tensors)
        if all_contiguous and numel_per > 16384:
            BLOCK_SIZE = 8192
            bpe = dtype.itemsize if hasattr(dtype, 'itemsize') else torch.tensor([], dtype=dtype).element_size()
            if bpe == 2:
                total_bytes = out.numel() * 2
                if total_bytes % 4 == 0:
                    work_out = out.view(torch.int32)
                    work_tensors = [t.contiguous().view(torch.int32) for t in tensors]
                else:
                    work_out = out
                    work_tensors = list(tensors)
            else:
                work_out = out
                work_tensors = list(tensors)

            nw = 4
            offset = 0
            i_t = 0
            while i_t < len(work_tensors):
                batch = work_tensors[i_t : i_t + 4]
                num = len(batch)
                sizes = []
                offsets_list = []
                ptrs = []
                for t in batch:
                    n = t.numel()
                    sizes.append(n)
                    offsets_list.append(offset)
                    ptrs.append(t)
                    offset += n
                while len(sizes) < 4:
                    sizes.append(0)
                    offsets_list.append(0)
                    ptrs.append(batch[0])
                max_size = max(sizes)
                num_programs = min(triton.cdiv(max_size, BLOCK_SIZE), 8)
                grid = (num_programs, num)
                cat_flat_kernel_4[grid](
                    work_out,
                    ptrs[0], ptrs[1], ptrs[2], ptrs[3],
                    offsets_list[0], offsets_list[1], offsets_list[2], offsets_list[3],
                    sizes[0], sizes[1], sizes[2], sizes[3],
                    num_programs,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=nw,
                )
                i_t += num
        else:
            dsts = [out.select(dim, i) for i in range(num_tensors)]
            torch._foreach_copy_(dsts, list(tensors))
    elif numel_per <= KERNEL_NUMEL_THRESHOLD:
        dsts = [out.select(dim, i) for i in range(num_tensors)]
        torch._foreach_copy_(dsts, list(tensors))
    else:
        dim_prod_post = 1
        for s in list(inp0_shape)[dim:]:
            dim_prod_post *= s

        BLOCK = 4096
        i = 0
        while i < num_tensors:
            batch = tensors[i:i + 4]
            num = len(batch)

            b_ptrs = []
            b_offsets = []
            b_totals = []
            for j, t in enumerate(batch):
                ct = t.contiguous()
                b_ptrs.append(ct)
                b_offsets.append(i + j)
                b_totals.append(ct.numel())
            while len(b_ptrs) < 4:
                b_ptrs.append(batch[0])
                b_offsets.append(0)
                b_totals.append(0)

            max_total = max(b_totals)
            num_programs = min(triton.cdiv(max_total, BLOCK), 48)
            grid = (num_programs, num)

            stack_general_kernel_4[grid](
                out,
                b_ptrs[0], b_ptrs[1], b_ptrs[2], b_ptrs[3],
                num_tensors * numel_per,
                dim_prod_post,
                num_tensors,
                b_offsets[0], b_offsets[1], b_offsets[2], b_offsets[3],
                b_totals[0], b_totals[1], b_totals[2], b_totals[3],
                BLOCK=BLOCK,
                num_warps=4,
            )
            i += num

    return out
