# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial

import torch

from flag_gems.runtime.configs_loader import TunedConfigLoader

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

SUPPORTED_OPS = ["mm", "bmm", "addmm", "attention"]

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
            "GEMS_SPACEMIT Invalid config for op_name=%s, arch_id=%s, dtype=%s. "
            "Changed from MICRO_M=%s, MICRO_N=%s, MICRO_K=%s to "
            "MICRO_M=%s, MICRO_K=%s, MICRO_N=%s",
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


def _config_pre_hook(configs, op_name, nargs):
    op_key = _strip_spacemit_suffix(op_name)
    arch_id = get_current_arch_id()
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
        legal_configs = LEGAL_CONFIGS[arch_id][op_key].get(torch.float32, [])
    if not legal_configs:
        return

    current_m = nargs.get("MICRO_M", 0)
    current_k = nargs.get("MICRO_K", 0)
    current_n = nargs.get("MICRO_N", 0)

    if "MICRO_M" not in nargs:
        # Kernel signature has no MICRO_* params: this is the
        # general kernel reached via intra-package imports
        # (e.g. baddbmm backward calls ops.bmm.bmm), not the
        # _spacemit gemm kernel. Injecting would crash the
        # launch with "Keyword argument MICRO_M ... unrecognised".
        return

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
        # FlagTree may call the first config's hook for the selected config.
        # All candidates share the same architecture/dtype micro tile.
        for config in configs:
            config.kwargs.update(fixed)
        logger.warning(
            "GEMS_SPACEMIT pre_hook fixed config for op=%s arch=%s dtype=%s: "
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

        if op_key in SUPPORTED_OPS and configs:
            # partial keeps the hook bound to the copied candidate list when
            # the autotuner deep-copies configs before pruning.
            pre_hook = partial(_config_pre_hook, configs, op_name)
            for config in configs:
                config.pre_hook = pre_hook

        return configs

    return _get_tuned_config


def prepare_tuned_configs(configs, nargs, **kwargs):
    # Autotune copies launch kwargs before calling each candidate's pre-hook.
    # Resolve dtype-dependent parameters before those copies are made.
    for config in configs:
        if config.pre_hook is not None:
            config.pre_hook({**nargs, **kwargs, **config.all_kwargs()})
    return [
        config
        for config in configs
        if all(
            config.kwargs[f"BLOCK_SIZE_{dim}"] % config.kwargs[f"MICRO_{dim}"] == 0
            for dim in ("M", "N", "K")
        )
    ]


def setup_triton_config():
    TunedConfigLoader.get_tuned_config = get_tuned_config(
        TunedConfigLoader.get_tuned_config
    )
