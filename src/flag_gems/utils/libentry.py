import hashlib
import inspect
import logging
import math
import os
import sqlite3
import threading
import time
import weakref
from abc import abstractmethod
from collections import OrderedDict
from itertools import starmap
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import triton

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend import vendor_module

from .code_cache import config_cache_dir

logger = logging.getLogger(__name__)

DEVICE_COUNT = runtime.device.device_count
ATTRS = {
    (2, 2): 5,
    (2, 3): 5,
    (3, 0): 4,
    (3, 1): 4,
    (3, 2): 4,
    (3, 3): 8,
}
# Set (3, 2) to 9 for cambricon (special Autotune config)
if vendor_module.vendor_info.vendor_name == "cambricon":
    ATTRS[(3, 2)] = 9

version = triton.__version__.split(".")
major_version, minor_version = eval(version[0]), eval(version[1])


if major_version == 2:

    def all_kwargs(self):
        return {
            **self.kwargs,
            **{
                k: getattr(self, k)
                for k in (
                    "num_warps",
                    "num_ctas",
                    "num_stages",
                    "num_buffers_warp_spec",
                    "num_consumer_groups",
                    "reg_dec_producer",
                    "reg_inc_consumer",
                    "maxnreg",
                )
                if hasattr(self, k)
            },
        }

    setattr(triton.Config, "all_kwargs", all_kwargs)


class LibCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LibCache, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.global_cache: Dict = {}
        self.volumn: Dict = {}
        self.cache_path = (
            config_cache_dir() / f"TunedConfig_{major_version}_{minor_version}.db"
        )
        self.preload()
        weakref.finalize(self, self.store)

    def __getitem__(self, key):
        if key not in self.global_cache:
            self.global_cache[key] = {}
        return self.global_cache[key]

    def preload(self):
        connect = sqlite3.connect(self.cache_path)
        c = connect.cursor()
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in c.fetchall()]
        for operator in tables:
            c.execute(
                f"CREATE TABLE IF NOT EXISTS {operator} (key TEXT PRIMARY KEY, config TEXT)"
            )
            cursor = c.execute(f"SELECT key, config from {operator}")
            cache = self.__getitem__(operator)

            for row in cursor:
                key_str, config_str = row
                key = [eval(k) for k in key_str[1:-1].split(", ")]

                cfg_ls = [item.split(": ") for item in config_str.split(", ")]
                kwargs = {}
                numargs = {}
                attrs = ATTRS[(major_version, minor_version)]
                for k, v in cfg_ls[:-attrs]:
                    kwargs[k] = eval(v)
                for k, v in cfg_ls[-attrs:]:
                    numargs[k] = eval(v)
                # In Triton v2.2 and v2.3, enable_persistent is stored in config cache
                # but not defined as initialization parameter
                numargs.pop("enable_persistent", None)
                config = triton.Config(kwargs, **numargs)
                cache[tuple(key)] = config
            self.volumn[operator] = len(cache)
        connect.close()

    def store(self):
        connect = sqlite3.connect(self.cache_path)
        c = connect.cursor()
        for operator, cache in self.global_cache.items():
            if len(cache) == self.volumn.get(operator, 0):
                continue

            c.execute(
                f"CREATE TABLE IF NOT EXISTS {operator} (key TEXT PRIMARY KEY, config TEXT)"
            )
            for key, config in cache.items():
                c.execute(
                    f"INSERT OR IGNORE INTO {operator} (key, config) VALUES (?, ?)",
                    (str(key), config.__str__()),
                )

        connect.commit()
        connect.close()


libcache = LibCache()


class LibTuner(triton.runtime.Autotuner):
    """`LibTuner` is the base class for `FlagGems` library autotuner.

    It could be extended in two ways, overriding the `policy` or `run` method in a subclass.
    For `policy` extension, `LibTuner` provides a decorator `register_policy` to register a policy function quickly.
    Please refer to the implementation of `default_policy` for an example.
    """

    # The dispatch table for `LibTuner` subclasses. It's shared across all instances.
    _dispatch_table: Dict[str, Type["LibTuner"]] = {}
    _strategy_table: Dict[str, Callable[[Any], Any]] = {}

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Optional[Dict] = None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        strategy=None,
    ):
        # NOTE(zhengyang): See discussion in https://github.com/triton-lang/triton/pull/4496
        if major_version == 2 or (major_version == 3 and minor_version <= 1):
            if warmup is None:
                warmup = 25
            if rep is None:
                rep = 100
        if major_version == 2:
            super().__init__(
                fn,
                arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                prune_configs_by,
                warmup,
                rep,
            )
            self.base_fn = fn
            while not inspect.isfunction(self.base_fn):
                self.base_fn = self.base_fn.fn
        else:
            super().__init__(
                fn,
                arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                pre_hook,
                post_hook,
                prune_configs_by,
                warmup,
                rep,
                use_cuda_graph,
            )
        self.__name__ = self.base_fn.__name__
        self.keys = key
        if isinstance(strategy, str):
            strategy = LibTuner.get_strategy(strategy)
        if not isinstance(strategy, (list, tuple)):
            strategy = [strategy] * len(self.keys)
        assert len(strategy) == len(
            self.keys
        ), f"the length of strategy {len(strategy)} must match the length of keys {len(self.keys)}"
        strategy: List[Callable[[Any], Any]] = [
            LibTuner.get_strategy(s) if isinstance(s, str) else s for s in strategy
        ]
        self.strategy = strategy
        # Use table name with hash instead of hash in key
        self.kernel_hash = None
        self.table_name = f"{self.__name__}_{self.get_kernel_hash()}"
        self.cache = libcache[self.table_name]

    def get_kernel_hash(self):
        if self.kernel_hash is None:
            jit_fn = self.fn
            while not isinstance(jit_fn, triton.runtime.JITFunction):
                jit_fn = jit_fn.fn
            func_hash = jit_fn.cache_key
            config_strs = [str(config) for config in self.configs]
            combined_content = f"{func_hash}{config_strs}"
            self.kernel_hash = hashlib.md5(combined_content.encode("utf-8")).hexdigest()
        return self.kernel_hash

    def get_key(self, args):
        if self.strategy is None:
            key = tuple(args[k] for k in self.keys if k in args)
        else:
            key = tuple(
                starmap(
                    lambda idx0, idx1: self.strategy[idx0](args[idx1]),
                    enumerate(self.keys),
                )
            )
        key += tuple(str(arg.dtype) for arg in args.values() if hasattr(arg, "dtype"))
        return key

    @staticmethod
    @abstractmethod
    def policy(
        self,
        fn: triton.runtime.KernelInterface,
        configs: Iterator[triton.Config],
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[triton.Config, Dict[str, float]]:
        raise NotImplementedError(
            f"`policy` isn't implemented in {self.__class__.__name__}"
        )

    @classmethod
    def register(cls, name: str):
        """Register a subclass of `LibTuner` with a name.

        Args:
            name: The name of the subclass.
        Returns:
            A decorator that registers the subclass with the name.
        """

        def decorator(subclass):
            cls._dispatch_table[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._dispatch_table[name]

    @classmethod
    def get_strategy(cls, name: str):
        return cls._strategy_table[name]

    @staticmethod
    def register_policy(
        name: str,
    ) -> Type["LibTuner"]:
        """A decorator to register a policy for `LibTuner`.

        This decorator allows you to create a new `LibTuner` subclass without defining a new class explicitly.
        The new subclass will have the `policy` method set to the provided policy function and will be registered under
        the specified name in the `LibTuner` dispatch table.
        """

        def decorator(
            policy_impl: Callable[
                [
                    triton.runtime.KernelInterface,
                    Iterator[triton.Config],
                    Tuple[Any],
                    Dict[str, Any],
                ],
                Tuple[triton.Config, Dict[str, float]],
            ],
        ):
            @LibTuner.register(name)
            class AnonymousLibTunerImpl(LibTuner):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def policy(
                    self,
                    fn: triton.runtime.KernelInterface,
                    configs: Iterator[triton.Config],
                    args: Tuple[Any],
                    kwargs: Dict[str, Any],
                ) -> Tuple[triton.Config, Dict[str, float]]:
                    return policy_impl(fn, configs, args, kwargs)

            return AnonymousLibTunerImpl

        return decorator

    @staticmethod
    def register_strategy(name: str):
        def decorator(
            strategy: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        ):
            LibTuner._strategy_table[name] = strategy
            return strategy

        return decorator

    def run(self, *args, **kwargs):
        # `arg_names` corresponds to the arguments of the `JITFunction`'s signature,
        # so please make sure the orders of `arg_names` and `args` match.
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for k, v in all_args.items() if k in self.arg_names}
            key = self.get_key(_args)
            if key not in self.cache:
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                best_config, timings = self.policy(
                    lambda config: self._bench(*args, config=config, **kwargs),
                    pruned_configs,
                    args,
                    kwargs,
                )
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = best_config
                full_nargs = {
                    **self.nargs,
                    **kwargs,
                    **self.cache[key].all_kwargs(),
                }
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret


@LibTuner.register_strategy(None)
@LibTuner.register_strategy("default")
def default_strategy(key: Any) -> Any:
    return key


@LibTuner.register_strategy("log")
def log2_strategy(key: Union[int, float]) -> float:
    return math.ceil(math.log2(key))


@LibTuner.register_policy("default")
def default_policy(
    bench_fn: triton.runtime.KernelInterface,
    configs: Iterator[triton.Config],
    args: Tuple[Any],
    kwargs: Dict[str, Any],
) -> Tuple[triton.Config, Dict[str, float]]:
    """Default policy for offline autotuning.

    Args:
        bench_fn: The function to benchmark.
        configs: The collection of the configuration search space.
        args: Kernel launch arguments.
        kwargs: Kernel launch arguments.
    Returns:
        A tuple containing the best configuration and a dictionary of timings for each configuration.

    This is one way to implement a default policy for offline autotuning. It's equal to the following
    ```
    @LibTuner.register("default")
    class DefaultLibTunerImpl(LibTuner):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )

        @staticmethod
        def policy(
            bench_fn: triton.runtime.KernelInterface,
            configs: Iterator[triton.Config],
            args: Tuple[Any],
            kwargs: Dict[str, Any],
        ) -> Tuple[triton.Config, Dict[str, float]]:
            timings: Dict[triton.Config, int] = {
                config: bench_fn(config) for config in configs
            }
            best_config: triton.Config = min(timings, key=timings.get)
            return best_config, timings
    ```
    In this way policies could be extended by registering a definition function quickly,
    or by creating a new subclass of `LibTuner` and overriding the `policy` method to have
    more control over the autotuning process.
    """
    timings: Dict[triton.Config, float] = {
        config: bench_fn(config) for config in configs
    }
    best_config: triton.Config = min(timings, key=timings.get)
    return best_config, timings


def libtuner(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=25,
    rep=100,
    use_cuda_graph=False,
    do_bench=None,
    strategy: Union[
        str, Callable[[Any], Any], List[Union[str, Callable[[Any], Any]]]
    ] = "default",
    policy: Union[str, Type[LibTuner]] = "default",
):
    """Decorator for triton library autotuner.

    `strategy` is a function that takes a key and returns a value.
    It accepts a string, which is the name of a registered strategy, or a callable function.
    In this form it will be applied to each key in the `key` list.
    If it's a tuple or list, it should have the same length as `key`,
    and each element should be a string or a callable function that takes a key and returns a value.
    `policy` accepts a string, which is the name of a registered `LibTuner` subclass, or a `LibTuner` subclass itself.
    """

    if isinstance(policy, str):
        policy = LibTuner.get(policy)
    assert issubclass(
        policy, LibTuner
    ), f"the class of {policy.__name__} is {policy.__class__.__name__}, not a subclass of {LibTuner.__name__}"

    def decorator(fn):
        return policy(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
            strategy=strategy,
        )

    return decorator


class LibEntry(triton.KernelInterface):
    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.divisibility = 16
        self.kernel_cache = tuple(dict() for _ in range(DEVICE_COUNT))

        while not isinstance(fn, triton.runtime.JITFunction):
            fn = fn.fn
        self.jit_function: triton.runtime.JITFunction = fn
        self.specialize_indices = [
            p.num
            for p in self.jit_function.params
            if not p.is_constexpr and not p.do_not_specialize
        ]
        self.do_not_specialize_indices = [
            p.num
            for p in self.jit_function.params
            if not p.is_constexpr and p.do_not_specialize
        ]
        self.lock = threading.Lock()
        self.signature = fn.signature

    def key(self, spec_args, dns_args, const_args):
        def spec_arg(arg):
            if hasattr(arg, "data_ptr"):
                return (arg.dtype, arg.data_ptr() % self.divisibility == 0)
            return (type(arg), arg)

        def dns_arg(arg):
            if hasattr(arg, "data_ptr"):
                return arg.dtype
            if not isinstance(arg, int):
                return type(arg)
            if -(2**31) <= arg and arg <= 2**31 - 1:
                return "i32"
            if 2**63 <= arg and arg <= 2**64 - 1:
                return "u64"
            return "i64"

        spec_key = [spec_arg(arg) for arg in spec_args]
        dns_key = [dns_arg(arg) for arg in dns_args]
        # const args passed by position
        return tuple(spec_key + dns_key + const_args)

    def run(self, *args, **kwargs):
        grid = kwargs["grid"]

        # collect all the arguments
        spec_args = []  # specialize arguments
        dns_args = []  # do not specialize arguments
        const_args = []  # constexpr arguments
        k_args = OrderedDict()
        param_names = list(self.signature.parameters.keys())
        for i, arg in enumerate(args):
            if i in self.specialize_indices:
                k_args[param_names[i]] = arg
                spec_args.append(arg)
            elif i in self.do_not_specialize_indices:
                k_args[param_names[i]] = arg
                dns_args.append(arg)
            else:
                if major_version == 3 and minor_version == 3:
                    k_args[param_names[i]] = arg
                const_args.append(arg)
        for p in self.jit_function.params[len(args) :]:
            if p.name in kwargs:
                val = kwargs[p.name]
            elif p.default is inspect._empty:
                continue
            else:
                val = p.default

            if p.is_constexpr:
                const_args.append(val)
                if major_version == 3 and minor_version == 3:
                    k_args[p.name] = val
            elif p.do_not_specialize:
                dns_args.append(val)
                k_args[p.name] = val
            else:
                spec_args.append(val)
                k_args[p.name] = val

        entry_key = self.key(spec_args, dns_args, const_args)
        device = torch_device_fn.current_device()
        cache = self.kernel_cache[device]
        while entry_key not in cache:
            # NOTE: we serialize the first run of a jit function regardless of which device to run on
            # because Triton runtime is currently not threadsafe.
            with self.lock:
                if entry_key in cache:
                    break
                kernel = self.fn.run(*args, **kwargs)
                fn = self.fn
                # collect constexpr arguments for grid computation
                constexprs = {}
                tune_constexprs = {}
                heur_constexprs = {}
                while not isinstance(fn, triton.runtime.JITFunction):
                    if isinstance(fn, triton.runtime.Autotuner):
                        config = fn.best_config
                        constexprs["num_warps"] = config.num_warps
                        constexprs["num_stages"] = config.num_stages
                        constexprs["num_ctas"] = config.num_ctas
                        constexprs = {**constexprs, **config.kwargs}
                        tune_constexprs = {**tune_constexprs, **config.kwargs}
                    elif isinstance(fn, triton.runtime.Heuristics):
                        for v, heur in fn.values.items():
                            heur_constexprs[v] = heur(
                                {
                                    **dict(zip(fn.arg_names, args)),
                                    **kwargs,
                                    **constexprs,
                                }
                            )
                            constexprs[v] = heur_constexprs[v]
                    else:
                        raise RuntimeError("Invalid Runtime Function")
                    fn = fn.fn
                for p in self.jit_function.params:
                    if (
                        p.is_constexpr
                        and p.name not in constexprs
                        and (p.default is not inspect._empty)
                    ):
                        constexprs[p.name] = p.default
                cache[entry_key] = (
                    kernel,
                    constexprs,
                    tune_constexprs,
                    heur_constexprs,
                )
            return kernel, constexprs

        kernel, constexprs, tune_constexprs, heur_constexprs = cache[entry_key]

        if callable(grid):
            # collect all arguments to the grid fn，ie:
            # 1. args,
            # 2. kwargs,
            # 3. all all other captured arguments in CompiledKernel from Autotunner & Heuristics
            # when kwargs & captured args conflict, captured args have higher priority
            meta = {**dict(zip(self.arg_names, args)), **kwargs, **constexprs}
            grid = grid(meta)
        grid = grid + (1, 1)

        if major_version == 3 and minor_version == 3:
            all_args = []
            missing_keys = []
            for key in list(self.signature.parameters.keys()):
                if key in k_args:
                    all_args.append(k_args[key])
                elif key in tune_constexprs:
                    all_args.append(tune_constexprs[key])
                elif key in heur_constexprs:
                    all_args.append(heur_constexprs[key])
                elif key in constexprs:
                    all_args.append(constexprs[key])
                else:
                    missing_keys.append(key)
                if len(missing_keys):
                    raise RuntimeError(
                        f"[libentry]: probably a bug, the following kernel params where not captured: {missing_keys}"
                    )
            kernel[grid[0:3]](*all_args)
        else:
            kernel[grid[0:3]](*k_args.values())
        return kernel, constexprs


def libentry():
    """Decorator for triton library entries."""

    def decorator(fn):
        return LibEntry(fn)

    return decorator
