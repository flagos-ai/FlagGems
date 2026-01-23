import warnings

from . import backend, common, error
from .backend.device import DeviceDetector


class Register:
    def __init__(
        self,
        config,
        user_include_ops_list=None,
        user_exclude_ops_list=None,
        cpp_patched_ops_list=None,
        lib=None,
    ):
        self.device = DeviceDetector()

        # lib is a instance of torch.library.Library
        # Some inference chips may not support the backward implementation of operators
        self.lib = lib

        # reg_key like 'CUDA', reg_bac_key like AutogradCUDA
        self.reg_key = self.device.dispatch_key
        self.all_ops = []

        if user_include_ops_list:
            self.include_ops = list(user_include_ops_list or [])
            self.exclude_ops = []
            self.config = config
            self.extract_include_config()
            # Use the filtered include config to avoid registering all ops.
            self.config = self.include_config
            self.for_each()
        else:
            self.vendor_unused_ops_list = self.get_vendor_unused_op()
            self.exclude_ops = (
                list(user_exclude_ops_list or []) + self.vendor_unused_ops_list
            )
            self.cpp_patched_ops_list = set(cpp_patched_ops_list or [])
            self.config = config
            self.config_filter()
            self.for_each()

    def extract_include_config(self):
        self.include_config = []
        for config_item in self.config:
            op_name = config_item[0]

            func = config_item[1]
            func_name = func.__name__ if hasattr(func, "__name__") else str(func)
            if func_name not in self.include_ops:
                continue

            if len(config_item) > 2:
                condition_func = config_item[2]
                if not condition_func():
                    continue

            self.include_config.append((op_name, config_item[1]))

        if not self.include_config:
            warnings.warn(
                "only_enable failed: No op to register. Check if include is correct."
            )
            return

    def config_filter(self):
        def enabled(item):
            return len(item) < 3 or bool(item[2]())

        self.config = [
            (item[0], item[1])
            for item in self.config
            if enabled(item)
            and item[1].__name__ not in self.exclude_ops
            and item[0] not in self.cpp_patched_ops_list
        ]

    def get_vendor_unused_op(self):
        if self.device.vendor != common.vendors.NVIDIA:
            return backend.get_curent_device_unused_op(self.device.vendor_name)
        return []

    def register_impl(self, key, fn):
        device_key = self.reg_key
        self.all_ops.append(key)
        self.lib.impl(key, fn, device_key)

    def for_each(self):
        try:
            for key, func in self.config:
                self.register_impl(key, func)
        except Exception as e:
            error.register_error(e)

    def get_all_ops(self):
        return self.all_ops

    def get_unused_ops(self):
        return self.exclude_ops

    def get_vendor_name(self):
        return self.device.vendor_name

    def get_current_device(self):
        return self.device.name
