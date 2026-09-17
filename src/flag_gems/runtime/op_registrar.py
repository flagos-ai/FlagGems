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

import warnings

from . import backend, common, error
from .backend.device_finder import DeviceDetector


class GeneralOpRegistrar:
    def __init__(
        self,
        config,
        user_include_ops=None,
        user_exclude_ops=None,
        cpp_patched_ops=None,
        lib=None,
        full_config_by_func=None,
    ):
        self.device = DeviceDetector()

        # lib is a instance of torch.library.Library
        # Some inference chips may not support the backward implementation of operators
        self.lib = lib

        # reg_key like 'CUDA'
        self.reg_key = self.device.dispatch_key
        self.all_ops = []
        self.all_keys = []
        if self.device.vendor == common.vendors.CAMBRICON:
            # TODO: Cambricon specific, to avoid op deadlock question in libtuner.
            # Should remove this in the future.
            self.torch_ops_map = {}

        # optional mapping func_name -> list of config entries
        self.full_config_by_func = full_config_by_func
        self.cpp_patched_ops = set(cpp_patched_ops or [])

        if user_include_ops:
            self.include_ops = list(user_include_ops or [])
            self.exclude_ops = []
            self.config = config
            self.extract_include_config()
            # Use the filtered include config to avoid registering all ops.
            self.config = self.include_config
            self.for_each()
        else:
            self.vendor_unused_ops_list = self.get_vendor_unused_op()
            self.exclude_ops = (
                list(user_exclude_ops or []) + self.vendor_unused_ops_list
            )
            self.config = config
            self.config_filter()
            self.for_each()

    def extract_include_config(self):
        # Simple fast path: if we have a full_config_by_func mapping, iterate
        # over the requested function names and collect matching config items.
        self.include_config = []

        if self.full_config_by_func:
            for name in self.include_ops:
                for config_item in self.full_config_by_func.get(name, []):
                    op_name, func = config_item[0], config_item[1]
                    # respect optional condition functions
                    if not self._config_enabled(config_item):
                        continue
                    if op_name in self.cpp_patched_ops:
                        continue
                    self.include_config.append(self._normalized_config(config_item))
        else:
            # fallback: scan provided config and match by func name or op name
            for config_item in self.config:
                op_name, func = config_item[0], config_item[1]
                func_name = func.__name__ if hasattr(func, "__name__") else str(func)
                if (
                    func_name not in self.include_ops
                    and op_name not in self.include_ops
                ):
                    continue
                if not self._config_enabled(config_item):
                    continue
                if op_name in self.cpp_patched_ops:
                    continue
                self.include_config.append(self._normalized_config(config_item))

        if not self.include_config:
            warnings.warn(
                "only_enable failed: No op to register. Check if include is correct."
            )
            return

    @staticmethod
    def _config_enabled(item):
        condition_func = item[2] if len(item) > 2 else None
        return condition_func is None or bool(condition_func())

    @staticmethod
    def _extra_dispatch_keys(item):
        return tuple(item[3]) if len(item) > 3 else ()

    def _normalized_config(self, item):
        key, fn = item[0], item[1]
        return (
            key,
            self._resolve_live_override(key, fn),
            self._extra_dispatch_keys(item),
        )

    def _resolve_live_override(self, key, fn):
        # Config entries capture a function reference at import time. If that
        # op has since been overridden on the flag_gems module (e.g. via
        # DynamicOpOverride), prefer the live attribute so registration picks
        # up the override instead of the stale reference.
        #
        # The dispatch key alone is ambiguous for naming purposes: several
        # overloads of one op (e.g. "_softmax" and "_softmax.out") share a
        # dispatch-key prefix but are bound to distinct functions (softmax
        # vs. softmax_out). To avoid colliding those onto the same module
        # attribute, look the key up in the authoritative _FULL_CONFIG to
        # find its originally-bound function, and resolve by *that*
        # function's own __name__.
        import sys

        module = sys.modules.get("flag_gems")
        if module is None:
            return fn

        has_full_config = self._module_has_full_config(module)
        original_func = self._original_func_for_key(module, key)

        if original_func is None and has_full_config:
            # _FULL_CONFIG is the authoritative source of registrable ops. If
            # it exists on the module but doesn't contain this key, the key
            # was never a real registration to begin with -- registering (or
            # overriding) it is not permitted.
            raise ValueError(
                f"Key '{key}' was not found in flag_gems._FULL_CONFIG; "
                "refusing to register/override an op that doesn't exist."
            )

        func_name = getattr(original_func, "__name__", None) if original_func else None

        if not func_name or func_name == "<lambda>":
            # No _FULL_CONFIG on the module at all (e.g. a synthetic config
            # passed directly, as in unit tests), so there's nothing
            # authoritative to check the key against. Fall back to deriving
            # a name from the key itself; this is only reached when there is
            # no _FULL_CONFIG to disambiguate against in the first place, so
            # the overload-collision risk above doesn't apply here.
            original_func = fn
            func_name = key.split(".", 1)[0]
            if func_name.startswith("_") and not func_name.startswith("__"):
                func_name = func_name[1:]

        current = getattr(module, func_name, None)
        if current is not None and current is not original_func:
            return current
        return fn

    @staticmethod
    def _module_has_full_config(module):
        return getattr(module, "_FULL_CONFIG", None) is not None

    def _original_func_for_key(self, module, key):
        key_to_func = getattr(self, "_full_config_key_to_func", None)
        if key_to_func is None:
            key_to_func = {}
            for entry in getattr(module, "_FULL_CONFIG", None) or ():
                if len(entry) >= 2:
                    key_to_func.setdefault(entry[0], entry[1])
            self._full_config_key_to_func = key_to_func
        return key_to_func.get(key)

    def config_filter(self):
        self.config = [
            self._normalized_config(item)
            for item in self.config
            if self._config_enabled(item)
            and item[1].__name__ not in self.exclude_ops
            and item[0] not in self.cpp_patched_ops
        ]

    def get_vendor_unused_op(self):
        return backend.get_unused_ops(self.device.vendor_name)

    def register_impl(self, key, fn, extra_dispatch_keys=()):
        if self.lib is None:
            raise ValueError("Library instance is not provided.")
        device_key = self.reg_key
        self.all_ops.append(fn.__name__)
        self.all_keys.append(key)
        if self.device.vendor == common.vendors.CAMBRICON:
            import torch

            try:
                self.torch_ops_map["aten::" + key] = torch.library.get_kernel(
                    "aten::" + key, device_key
                )
            except Exception:
                pass
            try:
                self.lib.impl(key, fn, device_key, allow_override=True)
            except TypeError:
                # Older torch versions don't support allow_override
                self.lib.impl(key, fn, device_key)
        else:
            self.lib.impl(key, fn, device_key)

        for dispatch_key in extra_dispatch_keys:
            self.lib.impl(key, fn, dispatch_key)

    def for_each(self):
        for key, func, extra_dispatch_keys in self.config:
            try:
                self.register_impl(key, func, extra_dispatch_keys)
            except Exception as e:
                error.register_error(e)

    def get_all_ops(self):
        return self.all_ops

    def get_all_keys(self):
        return self.all_keys

    def get_unused_ops(self):
        return self.exclude_ops

    def get_vendor_name(self):
        return self.device.vendor_name

    def get_current_device(self):
        return self.device.name
