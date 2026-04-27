import copy
import os
import warnings

import triton

from . import backend
from .backend.device import DeviceDetector

DEFAULT_EXPAND_CONFIG_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "utils",
        "configs",
        "general_ops_expand_configs.yaml",
    )
)


DEFAULT_STRATEGIES = {
    "bmm": ["log", "log", "log", "align32", "align32"],
    "addmm": ["align32", "align32", "align32"],
    "baddbmm": ["align32", "align32", "align32"],
    "mv": ["align32", "align32"],
    "w8a8_block_fp8_general": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
    ],
    "w8a8_block_fp8_general_splitk": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
    ],
    "w8a8_block_fp8_general_tma": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
        "default",
    ],
    "mm_general_tma": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
        "default",
    ],
    "gemv": ["align32", "align32", "align32", "default"],
    "sparse_attention": ["align32", "align32", "align32"],
}

OP_KEY_ORDERS = {
    "bmm": ["M", "N", "K", "stride_am", "stride_bk"],
    "addmm": ["M", "N", "K"],
    "baddbmm": ["M", "N", "K"],
    "mv": ["M", "N"],
    "w8a8_block_fp8_general": ["M", "N", "K", "stride_am", "stride_bk"],
    "w8a8_block_fp8_general_splitk": ["M", "N", "K", "stride_am", "stride_bk"],
    "w8a8_block_fp8_general_tma": ["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    "mm_general_tma": ["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    "gemv": ["M", "K", "stride_am", "stride_bk"],
    "sparse_attention": ["topk", "H_ACTUAL", "D"],
}


class ConfigLoader(object):
    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.device = DeviceDetector()
            # primitive_yaml_config is simply the dictionary returned by yaml
            # and is reserved from being an attr for vendor customizability
            self.arch_specialized_yaml_config = None
            self.arch_heuristics_config = None
            self.vendor_primitive_yaml_config = self.get_vendor_tune_config()
            self.default_primitive_yaml_config = self.get_default_tune_config()
            self.vendor_heuristics_config = self.get_vendor_heuristics_config()
            self.default_heuristics_config = self.get_default_heuristics_config()
            try:
                if backend.BackendArchEvent().has_arch:
                    self.arch_specialized_yaml_config = (
                        backend.BackendArchEvent().autotune_configs
                    )
                    self.arch_heuristics_config = (
                        backend.BackendArchEvent().heuristics_configs
                    )
            except Exception as err:
                print(f"[INFO] : {err}")

            if self.vendor_heuristics_config is None:
                vendorname = self.device.vendor_name
                warnings.warn(
                    f"The {vendorname} configuration of heuristics_config is None"
                )
            # gen_key is an identifier that indicates whether the current config needs to be generated automatically
            self.gen_key = "gen"
            # loaded_triton_config is wrapped in triton.Config according to primitive_yaml_config
            self.loaded_triton_config = {}
            self.triton_config_default = {
                "num_stages": 2,
                "num_warps": 4,
                "num_ctas": 1,
            }
            if self.device.vendor_name in ["hygon"]:
                self.triton_config_default = {
                    "num_stages": 2,
                    "num_warps": 4,
                    "num_ctas": 1,
                    "num_ldmatrixes": 0,
                }
            self.expand_config_registry = self._build_expand_registry()
            self.load_all()

    def _build_configs_by_op(self, op_name, ranges, pre_hook=None):
        if op_name == "bmm":
            return [
                triton.Config(
                    {
                        "TILE_M": block_m,
                        "TILE_N": block_n,
                        "TILE_K": block_k,
                        "GROUP_M": 1 if block_m == 32 else 2,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_n in ranges["BLOCK_N"]
                for block_k in ranges["BLOCK_K"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "addmm":
            return [
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "BLOCK_SIZE_K": block_k,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_n in ranges["BLOCK_N"]
                for block_k in ranges["BLOCK_K"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "baddbmm":
            return [
                triton.Config(
                    {
                        "TILE_M": block_m,
                        "TILE_N": block_n,
                        "TILE_K": block_k,
                        "GROUP_M": 1 if block_m <= 32 else 2,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_n in ranges["BLOCK_N"]
                for block_k in ranges["BLOCK_K"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "mv":
            return [
                triton.Config(
                    {
                        "BLOCK_N": block_n,
                        "BLOCK_M": block_m,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_n in ranges["BLOCK_N"]
                for block_m in ranges["BLOCK_M"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "mm_general_tma":
            return [
                triton.Config(
                    {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_n in ranges["BLOCK_N"]
                for block_k in ranges["BLOCK_K"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "gemv":
            return [
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_K": block_k},
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_k in ranges["BLOCK_K"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "sparse_attention":
            return [
                triton.Config(
                    {"BLOCK": block},
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block in ranges["BLOCK"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "w8a8_block_fp8_general":
            return [
                triton.Config(
                    {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_M": group_m,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_n in ranges["BLOCK_N"]
                for block_k in ranges["BLOCK_K"]
                for group_m in ranges["GROUP_M"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "w8a8_block_fp8_general_tma":
            group_m_values = ranges.get("GROUP_M", [None])
            return [
                triton.Config(
                    dict(
                        {
                            "BLOCK_M": block_m,
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                        },
                        **({} if group_m is None else {"GROUP_M": group_m}),
                    ),
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_n in ranges["BLOCK_N"]
                for block_k in ranges["BLOCK_K"]
                for group_m in group_m_values
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        if op_name == "w8a8_block_fp8_general_splitk":
            return [
                triton.Config(
                    {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "SPLIT_K": split_k,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for block_m in ranges["BLOCK_M"]
                for block_n in ranges["BLOCK_N"]
                for block_k in ranges["BLOCK_K"]
                for split_k in ranges["SPLIT_K"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]

        return []

    def _build_single_expand_spec(
        self,
        op_name,
        expand_yaml_path=None,
        yaml_op_name=None,
    ):
        return {
            "yaml_op_name": yaml_op_name or op_name,
            "key": OP_KEY_ORDERS[op_name],
            "default_strategy": DEFAULT_STRATEGIES[op_name],
            "expand_yaml_path": expand_yaml_path,
        }

    def _build_expand_registry(self):
        return {
            "bmm": self._build_single_expand_spec(
                "bmm", expand_yaml_path=DEFAULT_EXPAND_CONFIG_PATH
            ),
            "addmm": self._build_single_expand_spec(
                "addmm", expand_yaml_path=DEFAULT_EXPAND_CONFIG_PATH
            ),
            "baddbmm": self._build_single_expand_spec(
                "baddbmm", expand_yaml_path=DEFAULT_EXPAND_CONFIG_PATH
            ),
            "mv": self._build_single_expand_spec(
                "mv", expand_yaml_path=DEFAULT_EXPAND_CONFIG_PATH
            ),
            "w8a8_block_fp8_general": self._build_single_expand_spec(
                "w8a8_block_fp8_general"
            ),
            "w8a8_block_fp8_general_splitk": self._build_single_expand_spec(
                "w8a8_block_fp8_general_splitk"
            ),
            "w8a8_block_fp8_general_tma": self._build_single_expand_spec(
                "w8a8_block_fp8_general_tma"
            ),
            "mm_general_tma": self._build_single_expand_spec("mm_general_tma"),
            "gemv": self._build_single_expand_spec("gemv"),
            "sparse_attention": self._build_single_expand_spec("sparse_attention"),
        }

    def load_all(self):
        for key in self.vendor_primitive_yaml_config:
            self.loaded_triton_config[key] = self.get_tuned_config(key)

    def get_vendor_heuristics_config(self):
        return backend.get_heuristic_config(self.device.vendor_name)

    def get_default_heuristics_config(self):
        return backend.get_heuristic_config("nvidia")

    def get_default_tune_config(self):
        return backend.get_tune_config("nvidia")

    def get_vendor_tune_config(self):
        return backend.get_tune_config(self.device.vendor_name)

    def get_heuristics_config(self, op_name):
        if self.arch_heuristics_config and op_name in self.arch_heuristics_config:
            return self.arch_heuristics_config[op_name]
        elif op_name in self.vendor_heuristics_config:
            return self.vendor_heuristics_config[op_name]
        elif op_name in self.default_heuristics_config:
            return self.default_heuristics_config[op_name]
        else:
            warnings.warn(f"No heuristics config found for {op_name}")
            return None

    def _resolve_iteration_values(self, gen_config, config_var_key):
        if isinstance(config_var_key, (list, tuple)):
            return config_var_key
        if isinstance(config_var_key, int):
            return [config_var_key]
        return gen_config[config_var_key]

    def _gen_impl(
        self,
        gen_config,
        iteration_plan,
        std_config,
    ):
        all_configs = []
        final_step = len(iteration_plan)
        stack = [{"cur_config": std_config, "current_step": 0}]

        while stack:
            cur_state = stack[-1]
            stack.pop()
            cur_config = cur_state.get("cur_config")
            current_step = cur_state.get("current_step")

            if current_step == final_step:
                all_configs.append(
                    triton.Config(
                        cur_config["META"],
                        num_warps=cur_config["num_warps"],
                        num_stages=cur_config["num_stages"],
                        num_ctas=cur_config["num_ctas"],
                    )
                )
            else:
                cur_entry = iteration_plan[current_step]
                cur_key = cur_entry["key"]
                key_config = self._resolve_iteration_values(
                    gen_config, cur_entry["source"]
                )
                for single_value in key_config:
                    new_config = copy.deepcopy(cur_config)
                    if cur_entry["kind"] == "meta_field":
                        new_config["META"][cur_key] = single_value
                    elif cur_entry["kind"] == "meta_block":
                        new_config["META"] = copy.deepcopy(single_value)
                    else:
                        new_config[cur_key] = single_value
                    stack.append(
                        {
                            "cur_config": new_config,
                            "current_step": current_step + 1,
                        }
                    )
        return all_configs

    def to_gen_config(self, gen_config):
        param_config = gen_config["param_map"]
        meta_config = param_config["META"]
        iteration_plan = []

        if isinstance(meta_config, dict):
            for meta_key, source in meta_config.items():
                iteration_plan.append(
                    {"key": meta_key, "source": source, "kind": "meta_field"}
                )
        else:
            iteration_plan.append(
                {"key": "META", "source": meta_config, "kind": "meta_block"}
            )

        for key, source in param_config.items():
            if key == "META":
                continue
            iteration_plan.append(
                {"key": key, "source": source, "kind": "config_field"}
            )

        current_config = {"META": {}}
        current_config.update(self.triton_config_default)
        return self._gen_impl(
            gen_config,
            iteration_plan,
            current_config,
        )

    def get_expand_config(self, op_name, yaml_path=None):
        op_spec = self.expand_config_registry.get(op_name)
        if op_spec is None:
            return -1

        key = op_spec.get("key", [])
        default_strategy = op_spec.get("default_strategy")
        expand_yaml_path = op_spec.get("expand_yaml_path") or yaml_path
        yaml_op_name = op_spec.get("yaml_op_name", op_name)

        try:
            expand_configs = backend.get_expand_config(
                op_name=yaml_op_name,
                file_path=expand_yaml_path,
            )
            if not isinstance(expand_configs, list):
                return -1

            gen_config = None
            strategy_config = None
            for single_config in expand_configs:
                if isinstance(single_config, dict) and "param_map" in single_config:
                    gen_config = single_config

                if isinstance(single_config, dict) and "strategy" in single_config:
                    strategy_config = single_config.get("strategy")

            param_map = gen_config.get("param_map")
            meta_map = param_map.get("META")

            strategy = default_strategy
            if isinstance(strategy_config, dict):
                strategy = [
                    strategy_config.get(k, default_strategy[idx])
                    for idx, k in enumerate(key)
                ]

            ranges = {}

            for mapped_key in meta_map.values():
                ranges[mapped_key.upper()] = gen_config[mapped_key]
            ranges["s"] = gen_config[param_map.get("num_stages")]
            ranges["w"] = gen_config[param_map.get("num_warps")]

            return {
                "ranges": ranges,
                "strategy": strategy,
            }
        except Exception:
            return -1

    def ops_get_configs(self, op_name, yaml_path=None, pre_hook=None):
        expand_config = self.get_expand_config(op_name, yaml_path=yaml_path)
        if expand_config == -1:
            return []
        ranges = expand_config["ranges"]
        return self._build_configs_by_op(op_name, ranges, pre_hook=pre_hook)

    def get_tuned_config(self, op_name):
        if op_name in self.loaded_triton_config:
            return self.loaded_triton_config[op_name]

        if (
            self.arch_specialized_yaml_config
            and op_name in self.arch_specialized_yaml_config
        ):
            current_op_configs = self.arch_specialized_yaml_config[op_name]
        elif op_name in self.vendor_primitive_yaml_config:
            current_op_configs = self.vendor_primitive_yaml_config[op_name]
        else:
            current_op_configs = self.default_primitive_yaml_config[op_name]

        configs = []
        if len(current_op_configs) == 0:
            return configs

        for single_config in current_op_configs:
            if self.gen_key in single_config:
                configs.extend(self.to_gen_config(single_config))
                continue

            current_config = copy.deepcopy(self.triton_config_default)
            for default_param in current_config:
                if default_param in single_config:
                    current_config[default_param] = single_config[default_param]

            if self.device.vendor_name in ["hygon"]:
                configs.append(
                    triton.Config(
                        single_config["META"],
                        num_warps=current_config["num_warps"],
                        num_stages=current_config["num_stages"],
                        num_ctas=current_config["num_ctas"],
                        num_ldmatrixes=current_config["num_ldmatrixes"],
                    )
                )
            else:
                configs.append(
                    triton.Config(
                        single_config["META"],
                        num_warps=current_config["num_warps"],
                        num_stages=current_config["num_stages"],
                        num_ctas=current_config["num_ctas"],
                    )
                )
        return configs
