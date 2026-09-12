"""Hopper MM route predicates used by the shared FlagTune resolver."""

from typing import Any


def select_mm_route(a: Any, b: Any, module: Any) -> str:
    """Mirror the Hopper public ``mm`` dispatch using backend helpers."""
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    if a.shape[1] != b.shape[0]:
        return "invalid"
    m, k = a.shape
    _, n = b.shape
    torch = module.torch
    if m == 0 or n == 0:
        return "empty"
    c = torch.empty(
        (m, n), device=a.device, dtype=module.get_higher_dtype(a.dtype, b.dtype)
    )
    if n == 1:
        return "gemv"
    if module._splitk_gemv_scenario(
        a, b, c, m, n, k
    ) or module._batched_splitk_gemv_scenario(a, b, c, m, n, k):
        return "splitk_gemv"
    if module._select_warp_specialized_dispatch_plan(a, b, c, m, n, k) is not None:
        return "warp_specialized"
    if module.tma_transposed_direct_tuned_scenario(a, b, c, m, n, k):
        return "tma_transposed_direct"
    if module.tma_transposed_config(a, b, c, m, n, k) is not None:
        return "tma_transposed_splitk"
    if module.streamk_scenario(a, b, m, n, k):
        return "streamk"
    if module.splitk_scenario(a, b, m, n, k):
        if module._tma_splitk_two_step_config(a, b, c, m, n, k) is not None:
            return "tma_splitk_two_step"
        if c.dtype == torch.float32 and not torch.are_deterministic_algorithms_enabled():
            return "splitk"
        if c.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return "splitk_two_step"
    if hasattr(
        module.triton.tools.tensor_descriptor, "TensorDescriptor"
    ) and module.is_tma_compatible(a, b, n, k, c):
        return "general_tma"
    return "general"
