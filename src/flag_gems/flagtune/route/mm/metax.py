"""MetaX MM route predicates used by the shared FlagTune resolver."""

from typing import Any


def select_mm_route(a: Any, b: Any, module: Any) -> str:
    """Mirror the MetaX public ``mm`` dispatch using backend helpers."""
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
        return (
            "metax_gemv_k_parallel"
            if module._gemv_k_parallel_scenario(m, k)
            else "metax_gemv"
        )
    if module._small_n_mm_scenario(a, b, c, n, k):
        return "metax_small_n"
    if module._select_two_step_split_k(m, n, k) is not None:
        return "metax_splitk_two_step"
    nt_scenario = module.nt_mm_scenario(a, b, c, m, n, k)
    prefer_dense_nt = (
        c.dtype in (torch.float16, torch.bfloat16)
        and nt_scenario
        and module._prefer_dense_nt_over_generic_splitk(m, n, k)
    )
    if module.splitk_mm_scenario(m, n, k) and not prefer_dense_nt:
        if c.dtype == torch.float32 and not torch.are_deterministic_algorithms_enabled():
            return "metax_splitk"
        return "metax_splitk_two_step"
    if module.nn_mm_scenario(a, b, c, m, n, k):
        return "metax_nn"
    if nt_scenario:
        return "metax_nt"
    return "metax_general"
