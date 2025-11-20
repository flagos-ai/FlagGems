import copy
import warnings

import triton

from . import backend
from .backend.device import DeviceDetector
from .dispatcher import op_dispatcher


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
            self.vendor_primitive_yaml_config = self.get_vendor_tune_config()
            self.default_primitive_yaml_config = self.get_default_tune_config()
            self.vendor_heuristics_config = self.get_vendor_heuristics_config()
            self.default_heuristics_config = self.get_default_heuristics_config()

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
            self.load_all()

    def load_all(self):
        for key in self.vendor_primitive_yaml_config:
            self.loaded_triton_config[key] = self.get_tuned_config(key)

    def get_vendor_heuristics_config(self):
        if op_dispatcher.is_dispatch_configs:
            return op_dispatcher.configurations["heuristic"]
        else:
            return backend.get_heuristic_config(self.device.vendor_name)

    def get_default_heuristics_config(self):
        return backend.get_heuristic_config("nvidia")

    def get_default_tune_config(self):
        return backend.get_tune_config("nvidia")

    def get_vendor_tune_config(self):
        if op_dispatcher.is_dispatch_configs:
            return op_dispatcher.configurations["autotune"]
        else:
            return backend.get_tune_config(self.device.vendor_name)

    def get_heuristics_config(self, op_name):
        if op_name in self.vendor_heuristics_config:
            return self.vendor_heuristics_config[op_name]
        elif op_name in self.default_heuristics_config:
            return self.default_heuristics_config[op_name]
        else:
            warnings.warn(f"No heuristics config found for {op_name}")
            return None

    def _gen_impl(
        self,
        gen_config,
        param_config,
        iteration_keys,
        std_config,
    ):
        all_configs = []
        final_step = len(iteration_keys)
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
                cur_key = iteration_keys[current_step]
                if cur_key in param_config["META"]:
                    config_var_key = param_config["META"][cur_key]
                else:
                    config_var_key = param_config[cur_key]
                if isinstance(config_var_key, int):
                    key_config = [config_var_key]
                else:
                    key_config = gen_config[config_var_key]
                for single_value in key_config:
                    new_config = copy.deepcopy(cur_config)
                    if cur_key in param_config["META"]:
                        new_config["META"][cur_key] = single_value
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
        iteration_keys = list(meta_config) + list(param_config)
        iteration_keys.remove("META")
        current_config = {"META": {}}
        current_config.update(self.triton_config_default)
        return self._gen_impl(
            gen_config,
            param_config,
            iteration_keys,
            current_config,
        )

    def get_tuned_config(self, op_name):
        if op_name in self.loaded_triton_config:
            return self.loaded_triton_config[op_name]

        if op_name in self.vendor_primitive_yaml_config:
            current_op_configs = self.vendor_primitive_yaml_config[op_name]
        else:
            current_op_configs = self.default_primitive_yaml_config[op_name]

        configs = []
        if len(current_op_configs) == 0:
            return configs

        if len(current_op_configs) == 1:
            single_config = current_op_configs[0]
            if self.gen_key in single_config:
                return self.to_gen_config(single_config)

        for single_config in current_op_configs:
            current_config = self.triton_config_default
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
