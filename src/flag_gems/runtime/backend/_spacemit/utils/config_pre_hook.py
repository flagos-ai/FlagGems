import logging

import torch

from flag_gems.runtime.configloader import ConfigLoader

_GEMM_CONFIG_0 = {
    torch.float32: [{"MICRO_M": 8, "MICRO_N": 32, "MICRO_K": 32}],
    torch.float16: [{"MICRO_M": 16, "MICRO_N": 32, "MICRO_K": 8}],
}

_GEMM_CONFIG_A03C = {
    torch.float32: [{"MICRO_M": 8, "MICRO_N": 32, "MICRO_K": 32}],
    torch.float16: [{"MICRO_M": 8, "MICRO_N": 16, "MICRO_K": 16}],
}

_MM_CONFIG_A064 = {
    torch.float32: [{"MICRO_M": 8, "MICRO_N": 32, "MICRO_K": 32}],
    torch.float16: [{"MICRO_M": 16, "MICRO_N": 32, "MICRO_K": 8}],
}

_GEMM_CONFIG_F000 = {
    torch.float32: [{"MICRO_M": 8, "MICRO_N": 32, "MICRO_K": 32}],
    torch.float16: [{"MICRO_M": 32, "MICRO_N": 32, "MICRO_K": 2}],
}

LEGAL_CONFIGS = {
    "0x503C": {
        "mm": _GEMM_CONFIG_0,
        "bmm": _GEMM_CONFIG_0,
        "addmm": _GEMM_CONFIG_0,
        "attention": _GEMM_CONFIG_0,
    },
    "0xA03C": {
        "mm": _GEMM_CONFIG_A03C,
        "bmm": _GEMM_CONFIG_A03C,
        "addmm": _GEMM_CONFIG_A03C,
        "attention": _GEMM_CONFIG_A03C,
    },
    "0xA064": {
        "mm": _MM_CONFIG_A064,
        "bmm": _MM_CONFIG_A064,
        "addmm": _MM_CONFIG_A064,
        "attention": _MM_CONFIG_A064,
    },
    "0xF000": {
        "mm": _GEMM_CONFIG_F000,
        "bmm": _GEMM_CONFIG_F000,
        "addmm": _GEMM_CONFIG_F000,
        "attention": _GEMM_CONFIG_F000,
    },
}

SUPPORTED_OPS = ["mm", "bmm"]

logger = logging.getLogger(__name__)


def get_current_arch_id():
    import triton

    arch_id = triton.runtime.driver.active.current_arch_id
    return arch_id


def _strip_spacemit_suffix(op_name):
    """Remove _spacemit suffix from op_name if present."""
    if op_name.endswith("_spacemit"):
        return op_name[:-9]  # Remove "_spacemit" (9 characters)
    return op_name


def validate_and_fix_config(config, arch_id, op_name, dtype):
    # Remove _spacemit suffix for config lookup
    op_key = _strip_spacemit_suffix(op_name)

    if arch_id not in LEGAL_CONFIGS:
        return config

    if op_key not in LEGAL_CONFIGS[arch_id]:
        return config

    legal_configs = LEGAL_CONFIGS[arch_id][op_key].get(dtype, [])

    if not legal_configs:
        legal_configs = LEGAL_CONFIGS[arch_id][op_key].get(torch.float32, [])

    current_m = config.kwargs.get("MICRO_M", 0)
    current_k = config.kwargs.get("MICRO_K", 0)
    current_n = config.kwargs.get("MICRO_N", 0)

    is_legal = any(
        cfg["MICRO_M"] == current_m
        and cfg["MICRO_K"] == current_k
        and cfg["MICRO_N"] == current_n
        for cfg in legal_configs
    )

    if not is_legal:
        fixed_config = legal_configs[0]
        config.kwargs["MICRO_M"] = fixed_config["MICRO_M"]
        config.kwargs["MICRO_K"] = fixed_config["MICRO_K"]
        config.kwargs["MICRO_N"] = fixed_config["MICRO_N"]

        logger.warning(
            "Invalid config for op_name=%s, arch_id=%s, dtype=%s. Changed from "
            "MICRO_M=%s, MICRO_N=%s, MICRO_K=%s to MICRO_M=%s, MICRO_K=%s, MICRO_N=%s",
            op_name,
            arch_id,
            dtype,
            current_m,
            current_n,
            current_k,
            fixed_config["MICRO_M"],
            fixed_config["MICRO_K"],
            fixed_config["MICRO_N"],
        )

    return config


def get_tuned_config(func):
    def _get_tuned_config(self, op_name):
        configs = func(self, op_name)
        # filter out invalid configs that may be None or missing kwargs
        if isinstance(configs, list):
            configs = [
                c
                for c in configs
                if c is not None and getattr(c, "kwargs", None) is not None
            ]

        # Remove _spacemit suffix for config lookup
        op_key = _strip_spacemit_suffix(op_name)

        if op_key in SUPPORTED_OPS and configs and len(configs) > 0:
            arch_id = get_current_arch_id()

            def make_pre_hook(config_obj):
                def pre_hook(nargs):
                    # nargs is a dict mapping kernel arg names to values
                    dtype = None
                    for value in nargs.values():
                        if isinstance(value, torch.Tensor):
                            dtype = value.dtype
                            break

                    if arch_id not in LEGAL_CONFIGS:
                        return
                    if op_key not in LEGAL_CONFIGS[arch_id]:
                        return

                    legal_configs = LEGAL_CONFIGS[arch_id][op_key].get(dtype, [])
                    if not legal_configs:
                        legal_configs = LEGAL_CONFIGS[arch_id][op_key].get(
                            torch.float32, []
                        )
                    if not legal_configs:
                        return

                    current_m = nargs.get("MICRO_M", 0)
                    current_k = nargs.get("MICRO_K", 0)
                    current_n = nargs.get("MICRO_N", 0)

                    is_legal = any(
                        cfg["MICRO_M"] == current_m
                        and cfg["MICRO_K"] == current_k
                        and cfg["MICRO_N"] == current_n
                        for cfg in legal_configs
                    )

                    if not is_legal:
                        fixed = legal_configs[0]
                        # Fix nargs dict (used by LibEntry path)
                        nargs["MICRO_M"] = fixed["MICRO_M"]
                        nargs["MICRO_K"] = fixed["MICRO_K"]
                        nargs["MICRO_N"] = fixed["MICRO_N"]
                        # Fix config object kwargs (used by LibTuner path:
                        # config.all_kwargs() is called after pre_hook)
                        config_obj.kwargs["MICRO_M"] = fixed["MICRO_M"]
                        config_obj.kwargs["MICRO_K"] = fixed["MICRO_K"]
                        config_obj.kwargs["MICRO_N"] = fixed["MICRO_N"]
                        logger.warning(
                            "pre_hook fixed config for op=%s arch=%s dtype=%s: "
                            "MICRO_M=%s->%s, MICRO_K=%s->%s, MICRO_N=%s->%s",
                            op_name,
                            arch_id,
                            dtype,
                            current_m,
                            fixed["MICRO_M"],
                            current_k,
                            fixed["MICRO_K"],
                            current_n,
                            fixed["MICRO_N"],
                        )

                return pre_hook

            # ensure configs list non-empty before setting pre_hook
            if len(configs) > 0 and configs[0] is not None:
                configs[0].pre_hook = make_pre_hook(configs[0])

        return configs

    return _get_tuned_config


def setup_triton_config():
    ConfigLoader.get_tuned_config = get_tuned_config(ConfigLoader.get_tuned_config)
