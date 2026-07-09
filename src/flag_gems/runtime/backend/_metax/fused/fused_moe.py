import contextlib
import threading
from typing import Any

import torch

from flag_gems.fused import fused_moe as generic_fused_moe
from flag_gems.runtime.backend._metax.fused.moe_sum import moe_sum as metax_moe_sum

_PATCH_LOCK = threading.RLock()
_PLAIN_HALF_CONFIG_DTYPES = ("fp16", "bf16")
_DISABLE_DIRECT_SUM_MIN_TOKENS = 1 << 60


def _arg_value(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
    default: Any = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def _config_dtype_from_call(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    hidden_states = _arg_value(args, kwargs, "hidden_states", 0)
    if hidden_states is None:
        return None

    return generic_fused_moe._get_config_dtype_str(
        dtype=hidden_states.dtype,
        use_fp8_w8a8=_arg_value(args, kwargs, "use_fp8_w8a8", 8, False),
        use_int8_w8a16=_arg_value(args, kwargs, "use_int8_w8a16", 10, False),
        use_int4_w4a16=_arg_value(args, kwargs, "use_int4_w4a16", 11, False),
        ocp_mx_scheme=_arg_value(args, kwargs, "ocp_mx_scheme", 12),
    )


def _metax_adjust_moe_config(
    config: dict[str, Any],
    dtype: str | None,
    M: int | None = None,
    E: int | None = None,
    N: int | None = None,
    K: int | None = None,
    gemm_stage: str | None = None,
    enable_gemm_fast_path: bool = False,
) -> dict[str, Any]:
    if dtype not in _PLAIN_HALF_CONFIG_DTYPES:
        return config

    config = config.copy()

    block_k_cap = 32
    if M is not None and E is not None and N is not None:
        if M >= 4096:
            if N >= 4096:
                if gemm_stage == "gemm1":
                    block_k_cap = 16
                elif gemm_stage == "gemm2":
                    block_k_cap = 32 if K is not None and K >= 12288 else 16
                else:
                    block_k_cap = 16
            elif E <= 16:
                block_k_cap = 16

    if "BLOCK_SIZE_K" in config:
        if (
            gemm_stage == "gemm2"
            and block_k_cap == 32
            and N is not None
            and N >= 4096
            and K is not None
            and K >= 12288
        ):
            config["BLOCK_SIZE_K"] = 32
        else:
            config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], block_k_cap)

    if M is not None and M >= 4096 and E is not None and N is not None:
        if gemm_stage == "gemm1" and E <= 16 and N >= 2048:
            config["BLOCK_SIZE_N"] = 256
        elif gemm_stage == "gemm2" and N >= 256 and N % 256 == 0:
            config["BLOCK_SIZE_N"] = 256

        if N >= 4096:
            if E <= 16:
                config["GROUP_SIZE_M"] = 8
            elif E >= 128:
                config["GROUP_SIZE_M"] = 1
        elif E >= 128:
            config["GROUP_SIZE_M"] = 8
        elif gemm_stage == "gemm2" and E <= 16:
            config["GROUP_SIZE_M"] = 2

    if M is not None and M < 4096 and N is not None and N >= 4096:
        config["num_stages"] = min(config.get("num_stages", 2), 2)

    if (
        gemm_stage == "gemm1"
        and M is not None
        and M >= 4096
        and N is not None
        and enable_gemm_fast_path
    ):
        config["PAIR_GATE_UP_DOT"] = True

    return config


@contextlib.contextmanager
def _metax_moe_config_patch(disable_direct_sum: bool):
    with _PATCH_LOCK:
        original_get_default_config = generic_fused_moe.get_default_config
        original_try_get_optimal_moe_config = (
            generic_fused_moe.try_get_optimal_moe_config
        )
        original_direct_sum_min_tokens = generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS
        original_moe_sum = generic_fused_moe.moe_sum

        def metax_get_default_config(
            M: int,
            E: int,
            N: int,
            K: int,
            topk: int,
            dtype: str | None,
            block_shape: list[int] | None = None,
            gemm_stage: str = "gemm1",
            enable_gemm_fast_path: bool = False,
        ) -> dict[str, Any]:
            config = original_get_default_config(
                M,
                E,
                N,
                K,
                topk,
                dtype,
                block_shape,
                gemm_stage=gemm_stage,
                enable_gemm_fast_path=enable_gemm_fast_path,
            )
            return _metax_adjust_moe_config(
                config,
                dtype,
                M,
                E,
                N,
                K,
                gemm_stage,
                enable_gemm_fast_path,
            )

        def metax_try_get_optimal_moe_config(
            *args,
            **kwargs,
        ) -> dict[str, Any] | tuple[dict[str, Any], bool]:
            result = original_try_get_optimal_moe_config(*args, **kwargs)
            dtype = kwargs.get("dtype")
            if dtype is None and len(args) > 3:
                dtype = args[3]

            M = kwargs.get("M")
            if M is None and len(args) > 4:
                M = args[4]
            E = kwargs.get("E")
            if E is None and len(args) > 5:
                E = args[5]
            gemm_stage = kwargs.get("gemm_stage")
            if gemm_stage is None and len(args) > 7:
                gemm_stage = args[7]
            if gemm_stage is None:
                gemm_stage = "gemm1"
            enable_gemm_fast_path = kwargs.get("enable_gemm_fast_path")
            if enable_gemm_fast_path is None and len(args) > 8:
                enable_gemm_fast_path = args[8]
            enable_gemm_fast_path = bool(enable_gemm_fast_path)

            N = K = None
            if len(args) > 1:
                shape = args[0] if gemm_stage == "gemm1" else args[1]
                if len(shape) >= 3:
                    _, N, K = shape

            if isinstance(result, tuple):
                config, is_embedded = result
                config = _metax_adjust_moe_config(
                    config,
                    dtype,
                    M,
                    E,
                    N,
                    K,
                    gemm_stage,
                    enable_gemm_fast_path,
                )
                if dtype in _PLAIN_HALF_CONFIG_DTYPES:
                    is_embedded = False
                return config, is_embedded
            return _metax_adjust_moe_config(
                result,
                dtype,
                M,
                E,
                N,
                K,
                gemm_stage,
                enable_gemm_fast_path,
            )

        generic_fused_moe.get_default_config = metax_get_default_config
        generic_fused_moe.try_get_optimal_moe_config = metax_try_get_optimal_moe_config
        generic_fused_moe.moe_sum = metax_moe_sum
        if disable_direct_sum:
            generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = _DISABLE_DIRECT_SUM_MIN_TOKENS
        try:
            yield
        finally:
            generic_fused_moe.get_default_config = original_get_default_config
            generic_fused_moe.try_get_optimal_moe_config = (
                original_try_get_optimal_moe_config
            )
            generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = original_direct_sum_min_tokens
            generic_fused_moe.moe_sum = original_moe_sum


def fused_experts_impl(*args, **kwargs) -> torch.Tensor:
    config_dtype = _config_dtype_from_call(args, kwargs)
    with _metax_moe_config_patch(
        disable_direct_sum=config_dtype in _PLAIN_HALF_CONFIG_DTYPES
    ):
        return generic_fused_moe.fused_experts_impl(*args, **kwargs)


def inplace_fused_experts(*args, **kwargs) -> torch.Tensor:
    config_dtype = _config_dtype_from_call(args, kwargs)
    with _metax_moe_config_patch(
        disable_direct_sum=config_dtype in _PLAIN_HALF_CONFIG_DTYPES
    ):
        return generic_fused_moe.inplace_fused_experts(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs) -> torch.Tensor:
    config_dtype = _config_dtype_from_call(args, kwargs)
    with _metax_moe_config_patch(
        disable_direct_sum=config_dtype in _PLAIN_HALF_CONFIG_DTYPES
    ):
        return generic_fused_moe.outplace_fused_experts(*args, **kwargs)
