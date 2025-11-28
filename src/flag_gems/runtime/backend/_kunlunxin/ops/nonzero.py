import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def nonzero_kernel_heur_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["n_elements"], 12))  # cluster_num


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("nonzero"),
#     key=[
#         "n_elements",
#     ],
# )
@triton.heuristics(
    values={
        "BLOCK_SIZE": nonzero_kernel_heur_block_size,
    },
)
@triton.jit
def nonzero_kernel(
    inp,
    prefix_sum,
    out,
    n_elements: tl.constexpr,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask)
    out_offset = tl.load(prefix_sum + offset, mask=mask) - 1

    nonzero_mask = mask and inp_vals == True  # noqa

    idx_flat = offset
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        final_out_offset = out_offset * ndim + dim
        final_out_offset = tl.where(nonzero_mask, final_out_offset, -1)
        tl.store(out + final_out_offset, remainder, mask=nonzero_mask)


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS NONZERO")

    inp_ndim = inp.ndim

    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)

    inp_bool = inp_view
    if inp_view.dtype != torch.bool:
        inp_bool = inp_view != 0

    prefix_sum = inp_bool.cumsum(axis=0)

    num_nonzeros = n_elements
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    import os

    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"

    with torch_device_fn.device(inp.device):
        nonzero_kernel[grid](
            inp_bool,
            prefix_sum,
            out,
            n_elements,
            shape,
            inp_ndim,
            isCloseUnrollControl=True,
        )

    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]
    if "TRITONXPU_STORE_MASK_SIM" in os.environ:
        del os.environ["TRITONXPU_STORE_MASK_SIM"]

    num_nonzeros = prefix_sum[n_elements - 1].item()
    out = out[0:num_nonzeros]

    if as_tuple:
        return torch.unbind(out, dim=0)
    else:
        return out
