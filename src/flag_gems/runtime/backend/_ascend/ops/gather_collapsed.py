from typing import List, Tuple

import torch
import triton
import triton.language as tl

WARP_LIST = [8, 16, 32, 64]
REORDER_LIST = [True, False]
MEM_LIST = [120 * 1024, 216 * 1024]
BLOCK_SIZE_LIST = [32, 64, 128, 256, 512, 1024, 2048]


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


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={
                "BLOCK_SIZE": size,
                "shared_mem_dynamic_size": localmem,
                "enable_simt_reorder_instruction": is_reorder,
            },
            num_warps=warp,
        )
        for warp in WARP_LIST
        for localmem in MEM_LIST
        for size in BLOCK_SIZE_LIST
        for is_reorder in REORDER_LIST
    ],
    key=["num_elements"],
    warmup=25,
    rep=100,
)
@triton.jit
def gather_kernel_collapsed(
    inp_ptr,
    index_ptr,
    out_ptr,
    SIZE_OUTER,
    SIZE_DIM,
    SIZE_INNER,
    stride_inp_outer,
    stride_inp_dim,
    stride_inp_inner,
    stride_idx_outer,
    stride_idx_dim,
    stride_idx_inner,
    stride_out_outer,
    stride_out_dim,
    stride_out_inner,
    num_elements,
    with_negative_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    elements_per_prog = tl.cdiv(num_elements, num_programs)
    prog_start = pid * elements_per_prog
    prog_end = tl.minimum(prog_start + elements_per_prog, num_elements)

    for block_start in range(prog_start, prog_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < prog_end

        idx_val = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)

        if with_negative_index:
            idx_val = tl.where(idx_val < 0, idx_val + SIZE_DIM, idx_val)
        # Coordinate Reconstruction: (outer, dim, inner)
        off_inner = offsets % SIZE_INNER
        tmp = offsets // SIZE_INNER
        off_outer = tmp // SIZE_DIM

        # Input Offset Calculation
        inp_off = (
            off_outer * stride_inp_outer
            + idx_val * stride_inp_dim
            + off_inner * stride_inp_inner
        )
        val = tl.load(inp_ptr + inp_off, mask=mask, other=0.0)

        # Output Store
        tl.store(out_ptr + offsets, val, mask=mask)


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


def gather_collapsed(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    out: torch.Tensor,
    grid_fn,
    return_run_kernel: bool = True,
    with_negative_index=False,
):
    if out.shape != index.shape:
        raise ValueError(f"out.shape {out.shape} must equal index.shape {index.shape}")

    dim = normalize_dim(dim, inp.ndim)

    inp_3d, idx_3d, out_3d, SIZE_OUTER, SIZE_DIM, SIZE_INNER = _collapsed_3d_views(
        inp, dim, index, out
    )
    num_elements = out_3d.numel()

    def _run_kernel():
        gather_kernel_collapsed[grid_fn](
            inp_3d,
            idx_3d,
            out_3d,
            # Shapes
            SIZE_OUTER,
            SIZE_DIM,
            SIZE_INNER,
            # Strides
            inp_3d.stride(0),
            inp_3d.stride(1),
            inp_3d.stride(2),
            idx_3d.stride(0),
            idx_3d.stride(1),
            idx_3d.stride(2),
            out_3d.stride(0),
            out_3d.stride(1),
            out_3d.stride(2),
            # Meta
            num_elements,
            with_negative_index,
            force_simt_only=False,
        )

    if return_run_kernel:
        return _run_kernel

    _run_kernel()
