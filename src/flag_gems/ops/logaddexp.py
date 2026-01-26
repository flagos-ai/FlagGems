import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
def logaddexp_func(a, b):
    # compute in fp32 for stability
    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)

    m = tl.maximum(a32, b32)
    # stable: m + log(exp(a-m) + exp(b-m))
    out32 = m + tl.log(tl.exp(a32 - m) + tl.exp(b32 - m))

    # cast back to original dtype (FlagGems usually expects this)
    return out32.to(a.dtype)


def logaddexp(A, B):
    # A y B son tensores CUDA; FlagGems hará el dispatch
    return logaddexp_func(A, B)