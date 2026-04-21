import logging

import torch

from flag_gems.runtime.configloader import ConfigLoader

LEGAL_CONFIGS = {
    "0": {
        "mm": {
            torch.float32: [
                {"MICRO_M": 8, "MICRO_N": 16, "MICRO_K": 8},
            ],
            torch.float16: [
                {"MICRO_M": 8, "MICRO_N": 16, "MICRO_K": 16},
            ],
        }
    },
    "0x503C": {
        "mm": {
            torch.float32: [
                {"MICRO_M": 8, "MICRO_N": 16, "MICRO_K": 8},
            ],
            torch.float16: [
                {"MICRO_M": 8, "MICRO_N": 16, "MICRO_K": 16},
            ],
        }
    },
    "0xA03C": {
        "mm": {
            torch.float32: [
                {"MICRO_M": 8, "MICRO_N": 16, "MICRO_K": 8},
            ],
            torch.float16: [
                {"MICRO_M": 8, "MICRO_N": 16, "MICRO_K": 16},
            ],
        }
    },
    "0xA064": {
        "mm": {
            torch.float32: [
                {"MICRO_M": 8, "MICRO_N": 32, "MICRO_K": 32},
            ],
            torch.float16: [
                {"MICRO_M": 16, "MICRO_N": 32, "MICRO_K": 8},
            ],
        }
    },
}

SUPPORTED_OPS = ["mm"]

logger = logging.getLogger(__name__)


def get_current_arch_id():
    import triton

    arch_id = triton.runtime.driver.active.current_arch_id
    return arch_id


def validate_and_fix_config(config, arch_id, op_name, dtype):
    if arch_id not in LEGAL_CONFIGS:
        return config

    if op_name not in LEGAL_CONFIGS[arch_id]:
        return config

    legal_configs = LEGAL_CONFIGS[arch_id][op_name].get(dtype, [])

    if not legal_configs:
        legal_configs = LEGAL_CONFIGS[arch_id][op_name].get(torch.float32, [])

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
        if op_name in SUPPORTED_OPS and configs and len(configs) > 0:
            arch_id = get_current_arch_id()

            def pre_hook(*args, **kwargs):
                dtype = None
                if len(args) > 1 and isinstance(args[1], dict):
                    input_dict = args[1]
                    for value in input_dict.values():
                        if isinstance(value, torch.Tensor):
                            dtype = value.dtype
                            break

                if len(args) > 0 and hasattr(args[0], "kwargs"):
                    validate_and_fix_config(args[0], arch_id, op_name, dtype)

            configs[0].pre_hook = pre_hook

        return configs

    return _get_tuned_config


def setup_triton_config():
    ConfigLoader.get_tuned_config = get_tuned_config(ConfigLoader.get_tuned_config)
