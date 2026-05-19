import torch


def _try_import_vllm_extension(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _is_op_registered(lib_name, op_name):
    try:
        lib = getattr(torch.ops, lib_name)
        return hasattr(lib, op_name)
    except Exception:
        return False


def _ensure_vllm_library_exists(lib_name, ops_to_check=None):
    module_map = {
        "_C": "vllm._C",
        "_moe_C": "vllm._moe_C",
        "_vllm_fa3_C": "vllm.vllm_flash_attn._vllm_fa3_C",
        "_C_cache_ops": "vllm._C_cache_ops",
    }

    module_name = module_map.get(lib_name)
    if module_name:
        imported = _try_import_vllm_extension(module_name)
        if imported:
            if ops_to_check:
                for op_name in ops_to_check:
                    if _is_op_registered(lib_name, op_name):
                        return True
            else:
                return True

    return False


_LIB_OPS = {
    "_C": [
        "silu_and_mul",
        "cutlass_scaled_mm",
        "per_token_group_fp8_quant",
        "apply_repetition_penalties_",
    ],
    "_moe_C": ["topk_softmax", "moe_align_block_size", "grouped_topk", "moe_sum"],
    "_vllm_fa3_C": ["get_scheduler_metadata"],
    "_C_cache_ops": ["concat_and_cache_mla"],
}

_OP_SIGNATURES = {
    "_moe_C": {
        "topk_softmax": "(Tensor(a!) topk_weights, Tensor(b!) topk_indices, "
        "Tensor(c!) token_expert_indices, Tensor gating_output) -> ()",
        "moe_align_block_size": "(Tensor topk_ids, int num_experts, "
        "int block_size, Tensor(a!) sorted_token_ids, Tensor(b!) experts_ids, "
        "Tensor(c!) num_tokens_post_pad) -> ()",
        "grouped_topk": "(Tensor gating_output, int n_group, int topk_group, "
        "int topk, bool renormalize, float routed_scaling_factor, Tensor? bias, "
        "int scoring_func=0) -> (Tensor, Tensor, Tensor)",
        "moe_sum": "(Tensor input, Tensor(a!) output) -> ()",
    },
    "_C": {
        "silu_and_mul": "(Tensor(a!) out, Tensor input) -> ()",
        "cutlass_scaled_mm": "(Tensor(a!) out, Tensor input, Tensor weight, "
        "Tensor scale_a, Tensor scale_b, Tensor? bias=None) -> ()",
        "per_token_group_fp8_quant": "(Tensor input, Tensor(a!) output_q, "
        "Tensor(b!) output_s, int group_size, float eps, float fp8_min, "
        "float fp8_max, bool scale_ue8m0=False) -> ()",
        "apply_repetition_penalties_": "(Tensor(a!) logits, Tensor prompt_mask, "
        "Tensor output_mask, Tensor repetition_penalties) -> Tensor",
    },
    "_vllm_fa3_C": {
        "get_scheduler_metadata": "(int batch_size, int max_seqlen_q, int max_seqlen_k, "
        "int num_heads, int num_heads_k, int headdim, int headdim_v, "
        "ScalarType qkv_dtype, Tensor seqused_k, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? cu_seqlens_k_new=None, Tensor? seqused_q=None, "
        "Tensor? leftpad_k=None, int? page_size=None, "
        "int max_seqlen_k_new=0, bool is_causal=False, "
        "int window_size_left=-1, int window_size_right=-1, "
        "bool has_softcap=False, int num_splits=0, "
        "bool? pack_gqa=None, int sm_margin=0) -> Tensor",
    },
    "_C_cache_ops": {
        "concat_and_cache_mla": "(Tensor kv_c, Tensor k_pe, Tensor(a!) kv_cache, "
        "Tensor slot_mapping, str kv_cache_dtype, Tensor scale) -> ()",
    },
}


def _define_op_if_not_exists(lib_name, op_name, signature):
    if not _is_op_registered(lib_name, op_name):
        try:
            torch.library.define(f"{lib_name}::{op_name}", signature)
        except Exception as e:
            print(f"Warning: Failed to define {lib_name}::{op_name}: {e}")


_libs_loaded = {}
for lib_name, ops in _LIB_OPS.items():
    loaded = _ensure_vllm_library_exists(lib_name, ops)
    _libs_loaded[lib_name] = loaded

    # looks like it happens only when vllm is not compiled
    # with custom ops like tsingmicro vllm
    if not loaded:
        if lib_name in _OP_SIGNATURES:
            for op_name, signature in _OP_SIGNATURES[lib_name].items():
                _define_op_if_not_exists(lib_name, op_name, signature)

vllm_C_lib = torch.library.Library("_C", "IMPL")
vllm_moe_C_lib = torch.library.Library("_moe_C", "IMPL")
vllm_fa3_C_lib = torch.library.Library("_vllm_fa3_C", "IMPL")
vllm_C_cache_ops_lib = torch.library.Library("_C_cache_ops", "IMPL")

libs = {
    "_C": vllm_C_lib,
    "_moe_C": vllm_moe_C_lib,
    "_vllm_fa3_C": vllm_fa3_C_lib,
    "_C_cache_ops": vllm_C_cache_ops_lib,
}


def patch_module_method(cls, method_name: str, new_method: callable, verbose=True):
    old_method = getattr(cls, method_name, None)
    setattr(cls, method_name, new_method)
    if verbose:
        print(
            f"Patched {cls.__name__}.{method_name} with FLAGGEMS {new_method.__name__}"
        )
    return old_method


def patch_vllm_lib(lib_name, fn_name, fn, key, verbose=True):
    if lib_name not in libs:
        raise ValueError(f"Library {lib_name} is not recognized.")

    lib = libs[lib_name]
    lib.impl(fn_name, fn, key)

    if verbose:
        print(f"Patched torch.ops.{lib_name}.{fn_name} with FLAGGEMS {fn.__name__}")
