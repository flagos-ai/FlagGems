"""Monkey-patch Qwen3_5GatedDeltaNet.recurrent_gated_delta_rule with a
TLE-CPU fused decode kernel.

Replaces the per-layer ATen sequence

    state *= exp(g)
    kv_mem  = (state * k.unsqueeze(-1)).sum(dim=-2)
    delta   = (v - kv_mem) * beta
    state  += k.unsqueeze(-1) * delta.unsqueeze(-2)
    out     = (state * q.unsqueeze(-1)).sum(dim=-2)

with a single fused @triton.jit kernel that calls the TLE builtin
`tle_ops.gated_delta_decode`, which dispatches to the NEON C runtime
`standalone_gated_delta_decode_fp32`.

State update + output dot are fused into a single sweep over state, matching
llama.cpp's Metal/SYCL backends (their CPU kernel does these as 2 separate
passes).

Decode (T=1) only — prefill (T>1) falls back to torch_chunk_gated_delta_rule.
"""

import logging

import torch
import triton
import triton.language as tl

from triton.language.extra.cpu.tle_ops import gated_delta_decode as _tle_gated_delta_decode

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit
def _gated_delta_decode_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    state_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr,
    k_dim: tl.constexpr, v_dim: tl.constexpr,
    use_l2norm: tl.constexpr,
):
    _tle_gated_delta_decode(
        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
        state_ptr, out_ptr,
        B, H, k_dim, v_dim, use_l2norm,
    )


def _patched_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """Drop-in for torch_recurrent_gated_delta_rule on T=1 decode path.

    Shapes (matching HF):
      query, key:    [B, T, H, k_dim] (any dtype; cast to fp32 internally)
      value:         [B, T, H, v_dim]
      g, beta:       [B, T, H]
      initial_state: [B, H, k_dim, v_dim] fp32, or None

    Returns:
      core_attn_out:        [B, T, H, v_dim] cast back to query.dtype
      last_recurrent_state: [B, H, k_dim, v_dim] fp32 (or None)

    For T>1, falls back to the original torch implementation in the host
    module (caller passes that in via the closure during patching).
    """
    raise NotImplementedError("install via patch_qwen3_5_gated_delta(model)")


def _make_patched_fn(torch_recurrent_fn):
    def fn(query, key, value, g, beta,
           initial_state, output_final_state,
           use_qk_l2norm_in_kernel=False):
        B, T, H, k_dim = query.shape
        v_dim = value.shape[-1]

        # Prefill or any non-decode shape: defer to the torch reference.
        if T != 1 or k_dim > 256 or v_dim > 256 or k_dim % 4 != 0 or v_dim % 4 != 0:
            return torch_recurrent_fn(query, key, value, g, beta,
                                      initial_state, output_final_state,
                                      use_qk_l2norm_in_kernel)

        orig_dtype = query.dtype

        # Squeeze T=1; cast to fp32 contiguous flat tensors that our kernel
        # expects ([B, H, k_dim] / [B, H, v_dim] / [B, H]).
        q_f = query.squeeze(1).to(torch.float32).contiguous()
        k_f = key.squeeze(1).to(torch.float32).contiguous()
        v_f = value.squeeze(1).to(torch.float32).contiguous()
        g_f = g.squeeze(1).to(torch.float32).contiguous()
        b_f = beta.squeeze(1).to(torch.float32).contiguous()

        if initial_state is None:
            state = torch.zeros(B, H, k_dim, v_dim, dtype=torch.float32)
        else:
            # .contiguous() on already-contiguous fp32 is a no-op.
            # The caller replaces cache_params.recurrent_states[layer_idx]
            # with our return value, so in-place update is safe here.
            state = initial_state.to(torch.float32).contiguous()

        out = torch.empty(B, H, v_dim, dtype=torch.float32)

        _gated_delta_decode_kernel[(1,)](
            q_f, k_f, v_f, g_f, b_f, state, out,
            B=B, H=H, k_dim=k_dim, v_dim=v_dim,
            use_l2norm=1 if use_qk_l2norm_in_kernel else 0,
        )

        core_attn_out = out.unsqueeze(1).to(orig_dtype).contiguous()
        last_recurrent_state = state if output_final_state else None
        return core_attn_out, last_recurrent_state
    return fn


def _get_qwen3_5_gated_delta_classes() -> tuple:
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5GatedDeltaNet"),
        ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeGatedDeltaNet"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextGatedDeltaNet"),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_gated_delta(model) -> int:
    """Replace each GDN module's recurrent_gated_delta_rule with the fused
    TLE kernel. Returns the number of modules patched.

    Safe to call multiple times (each module is patched once via id-tracking).
    """
    gdn_classes = _get_qwen3_5_gated_delta_classes()
    if not gdn_classes:
        logger.debug("No Qwen GDN classes found in transformers, skipping patch")
        return 0

    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) not in _PATCHED:
            torch_recurrent_fn = module.recurrent_gated_delta_rule
            module._original_recurrent_gated_delta_rule = torch_recurrent_fn
            module.recurrent_gated_delta_rule = _make_patched_fn(torch_recurrent_fn)
            _PATCHED.add(id(module))
            n += 1
    if n > 0:
        cls_names = ", ".join(c.__name__ for c in gdn_classes)
        logger.info(
            "Patched %d GDN modules (classes: %s) with TLE gated_delta_decode",
            n, cls_names,
        )
    return n


def unpatch_qwen3_5_gated_delta(model) -> int:
    gdn_classes = _get_qwen3_5_gated_delta_classes()
    if not gdn_classes:
        return 0
    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) in _PATCHED:
            if hasattr(module, "_original_recurrent_gated_delta_rule"):
                module.recurrent_gated_delta_rule = module._original_recurrent_gated_delta_rule
                del module._original_recurrent_gated_delta_rule
            _PATCHED.discard(id(module))
            n += 1
    return n
