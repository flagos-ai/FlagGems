import warnings
import functools
import torch

from . import backend, common, error
from .backend.device import DeviceDetector
from ..logging_utils import (
    precision_config,
    get_precision_logger,
    get_tensor_info,
    get_call_location,
    compare_outputs,
)


class Register:
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
                    if len(config_item) > 2:
                        condition_func = config_item[2]
                        if not condition_func():
                            continue
                    if op_name in self.cpp_patched_ops:
                        continue
                    self.include_config.append((op_name, func))
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
                if len(config_item) > 2:
                    condition_func = config_item[2]
                    if not condition_func():
                        continue
                if op_name in self.cpp_patched_ops:
                    continue
                self.include_config.append((op_name, func))

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
            and item[0] not in self.cpp_patched_ops
        ]

    def get_vendor_unused_op(self):
        if self.device.vendor != common.vendors.NVIDIA:
            return backend.get_curent_device_unused_op(self.device.vendor_name)
        return []

    def register_impl(self, key, fn):
        if self.lib is None:
            raise ValueError("Library instance is not provided.")
        device_key = self.reg_key
        self.all_ops.append(fn.__name__)
        self.all_keys.append(key)
        self.lib.impl(key, fn, device_key)

    def for_each(self):
        for key, func in self.config:
            try:
                self.register_impl(key, func)
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

# def _wrap_op_with_precision_check(op_key, fn):
#     @functools.wraps(fn)
#     def wrapper(*args, **kwargs):
#         fg_result = fn(*args, **kwargs)
        
#         cfg = precision_config
#         if cfg["enabled"] and op_key not in cfg["logged_ops"]:
#             cfg["logged_ops"].add(op_key)
#             logger = get_precision_logger()
#             input_info = [get_tensor_info(a) for a in args if get_tensor_info(a)]
#             logger.info(f"Op: {op_key} | in: {input_info}")
        
#         return fg_result
#     return wrapper

# 精度检查时允许的最大 tensor 元素数（超过则跳过，避免大 tensor 拷贝开销）
_MAX_NUMEL_FOR_CHECK = 1 * 1024 * 1024  # 1M 元素


def _get_dtype_tolerance(args, default_rtol, default_atol):
    """根据输入 tensor 的 dtype 自动调整容差"""
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            if a.dtype in (torch.bfloat16, torch.float16):
                return (max(default_rtol, 1e-2), max(default_atol, 1e-2))
            break
    return (default_rtol, default_atol)


def _to_cpu(x):
    """递归地将 tensor 移到 CPU"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(i) for i in x)
    elif isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x


def _max_tensor_numel(args):
    """返回参数中最大 tensor 的元素数"""
    max_n = 0
    for a in args:
        if isinstance(a, torch.Tensor):
            max_n = max(max_n, a.numel())
    return max_n


def _wrap_op_with_precision_check(op_key, fn):
    """包装 FlagGems 算子，在执行后与 PyTorch 原生 CPU 实现做精度对比。

    由于 FlagGems 替换了 CUDA dispatch，无法在 GPU 上调用原生实现，
    因此将输入拷贝到 CPU 计算参考结果。通过以下方式控制性能开销：
    - max_checks: 每个算子只检查前 N 次调用（默认 10）
    - 跳过大 tensor（超过 1M 元素）
    - 一旦记录过失败就不再检查该算子
    """
    _call_count = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal _call_count

        # 先执行 FlagGems 实现
        fg_result = fn(*args, **kwargs)

        cfg = precision_config
        if not cfg["enabled"]:
            return fg_result

        # 已经记录过失败的算子不再检查
        if op_key in cfg["logged_ops"]:
            return fg_result

        # 采样：每个算子只检查前 N 次调用
        _call_count += 1
        if _call_count > cfg.get("max_checks", 10):
            return fg_result

        op_name = op_key.split("::")[-1].split(".")[0] if "::" in op_key else op_key.split(".")[0]

        # 跳过 out variant（参数签名不同，CPU 调用容易出错）
        overload_part = op_key.split(".")[-1] if "." in op_key else ""
        if overload_part == "out" or op_name.endswith("_out"):
            return fg_result

        # 跳过不需要检查的算子
        skip_ops = {
            # 纯 layout / 内存操作
            'copy_', '_to_copy', 'view', 'reshape', 'expand', 'permute',
            'transpose', 'contiguous', 'clone', 'to', 'empty', 'zeros',
            'ones', 'full', 'masked_fill_',
            # 随机采样算子（GPU/CPU 随机数生成器不同，结果必然不一致）
            'exponential_', 'normal_', 'uniform_', 'bernoulli_', 'random_',
            'multinomial', 'randperm',
            # 排序/选择算子（相同值的排序顺序可能不同，不适合逐元素比较）
            'sort', 'topk', 'argsort',
        }
        if op_name in skip_ops:
            return fg_result

        # 跳过大 tensor，避免拷贝开销
        if _max_tensor_numel(args) > _MAX_NUMEL_FOR_CHECK:
            return fg_result

        try:
            # 解析 op_key 获取正确的 overload
            # op_key 形如 "add.Tensor", "mm.default", "softmax.int" 等
            parts = op_key.split(".")
            base_name = parts[0]
            overload_name = parts[1] if len(parts) > 1 else "default"

            aten_packet = getattr(torch.ops.aten, base_name, None)
            if aten_packet is None:
                return fg_result
            aten_overload = getattr(aten_packet, overload_name, None)
            if aten_overload is None:
                return fg_result

            # 将输入拷贝到 CPU，调用原生 aten 实现（CPU 上不受 FlagGems 影响）
            cpu_args = [_to_cpu(a) for a in args]
            cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}

            with torch.no_grad():
                pt_result_cpu = aten_overload(*cpu_args, **cpu_kwargs)

            # 将 FlagGems 结果也拷贝到 CPU 做比较（避免把 CPU 结果搬回 GPU）
            fg_result_cpu = _to_cpu(fg_result)

            # 根据 dtype 自动调整容差
            rtol, atol = _get_dtype_tolerance(args, cfg["rtol"], cfg["atol"])

            is_close, info = compare_outputs(fg_result_cpu, pt_result_cpu, rtol, atol)

            if not is_close:
                cfg["logged_ops"].add(op_key)
                logger = get_precision_logger()
                input_info = [get_tensor_info(a) for a in args if get_tensor_info(a)]
                output_info = get_tensor_info(fg_result)

                msg = f"Op: {op_key} | FAIL | in: {input_info} | out: {output_info}"
                if "error" in info:
                    msg += f" | {info['error']}: fg={info['fg']}, pt={info['pt']}"
                else:
                    msg += f" | max_abs: {info['max_abs']:.6e} | max_rel: {info['max_rel']:.6e}"
                msg += f" | rtol={rtol}, atol={atol}"
                logger.warning(msg)

        except Exception:
            pass

        return fg_result

    return wrapper

class PrecisionCheckRegister(Register):
    def register_impl(self, key, fn):
        if self.lib is None:
            raise ValueError("Library instance is not provided.")

        wrapped_fn = _wrap_op_with_precision_check(key, fn)

        device_key = self.reg_key
        self.all_ops.append(fn.__name__)
        self.all_keys.append(key)
        self.lib.impl(key, wrapped_fn, device_key)
