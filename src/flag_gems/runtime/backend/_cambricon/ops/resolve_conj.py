import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def conj_func(x):
    return x ^ (1 << 63)


def resolve_conj(A: torch.Tensor):
    logging.debug("GEMS_CAMBRICON RESOLVE_CONJ")
    assert (
        A.dtype == torch.cfloat
    ), "The `resolve_conj` operation in FlagGems currently only supports the `torch.cfloat` type"
    if A.is_conj():
        typed_view = torch.view_as_real(A.conj()).view(torch.int64)
        out = conj_func(typed_view)
        return torch.view_as_complex(out.view(torch.float))
    else:
        return A
