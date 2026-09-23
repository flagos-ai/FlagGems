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

from backend_utils import VendorDescriptor  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="kunlunxin",
    device_name="cuda",
    device_query_cmd="xpu-smi",
    triton_extra_name="xpu",
    fp64_enabled=False,
    fp8_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "atan2_out",
    "cumsum",
    "grid_sampler_3d_backward",
    "randperm",
    "searchsorted",
    "searchsorted_out",
    "searchsorted_scalar",
    "searchsorted_scalar_out",
    "topk",
    "unique",
    "slice",
    "conv_transpose1d",
    "mkldnn_rnn_layer",
    "_linalg_eigvals",
    "linalg_eig",
    "linalg_eigvals",
    "linalg_eigvals.out",
    "linalg_eigvals_out",
)


def _install_register_config_patch():
    """Register ops that are missing from the generic _FULL_CONFIG.

    The generic src/flag_gems/ops/__init__.py registers ("atanh", atanh) but
    NOT ("atanh_", atanh_), so torch.atanh_ under use_gems still falls through
    to the native xdnn atanh_, which raises
        [NOT IMPLEMENTED] ... scalar type of ret : kbfloat16 is unsupported
    for bf16 input (6 functional failures in tests/test_atanh_.py, fp16/fp32
    pass because native supports them). :func:`flag_gems.ops.atanh.atanh_`
    exists and handles bf16, it just was never registered for dispatch.

    Same mechanism as runtime/backend/_sunrise/__init__.py: extend the config
    passed to GeneralOpRegistrar with the missing entries (kept scoped inside
    use_gems / enable_gems; nothing is registered outside them).
    """

    from flag_gems.runtime.op_registrar import GeneralOpRegistrar

    register_cls = GeneralOpRegistrar

    if getattr(register_cls, "_kunlunxin_config_patched", False):
        return

    original_init = register_cls.__init__

    def _kunlunxin_extra_config_entries():
        from .ops.atanh import atanh_

        return (("atanh_", atanh_),)

    def _extend_config(config, full_config_by_func):
        extra_entries = _kunlunxin_extra_config_entries()
        existing_keys = {item[0] for item in config}
        merged_config = tuple(config) + tuple(
            item for item in extra_entries if item[0] not in existing_keys
        )

        if full_config_by_func is None:
            return merged_config, None

        merged_map = {key: list(value) for key, value in full_config_by_func.items()}
        for item in extra_entries:
            fn = item[1]
            func_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
            merged_map.setdefault(func_name, [])
            if item not in merged_map[func_name]:
                merged_map[func_name].append(item)
        return merged_config, merged_map

    def __init__(
        self,
        config,
        user_include_ops=None,
        user_exclude_ops=None,
        cpp_patched_ops=None,
        lib=None,
        full_config_by_func=None,
    ):
        config, full_config_by_func = _extend_config(config, full_config_by_func)
        return original_init(
            self,
            config,
            user_include_ops=user_include_ops,
            user_exclude_ops=user_exclude_ops,
            cpp_patched_ops=cpp_patched_ops,
            lib=lib,
            full_config_by_func=full_config_by_func,
        )

    register_cls.__init__ = __init__
    register_cls._kunlunxin_config_patched = True


_install_register_config_patch()


__all__ = ["*"]
