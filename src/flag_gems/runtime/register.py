from . import backend, commom_utils, error
from .backend.device import DeviceDetector


class Register:
    def __init__(
        self, config, user_unused_ops_list=None, cpp_patched_ops_list=None, lib=None
    ):
        # lib is a instance of torch.library.Library
        self.device = DeviceDetector()

        # Some inference chips may not support the backward implementation of operators
        self.lib = lib

        # reg_key like 'CUDA', reg_bac_key like AutogradCUDA
        self.reg_key = self.device.dispatch_key

        self.all_ops = []
        self.vendor_unused_ops_list = self.get_vendor_unused_op()
        self.unused_ops = list(user_unused_ops_list or []) + self.vendor_unused_ops_list
        self.cpp_patched_ops_list = set(cpp_patched_ops_list or [])
        self.config = config
        self.config_filter()
        self.for_each()

    def config_filter(self):
        self.config = [
            item
            for item in self.config
            if item[1].__name__ not in self.unused_ops
            and item[0] not in self.cpp_patched_ops_list
        ]

    def get_vendor_unused_op(self):
        if self.device.vendor != commom_utils.vendors.NVIDIA:
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
        return self.unused_ops

    def get_vendor_name(self):
        return self.device.vendor_name

    def get_current_device(self):
        return self.device.name
