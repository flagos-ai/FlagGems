import builtins
import inspect
import math
import os
import sqlite3
import threading
import time
import warnings
import weakref
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import triton

from .. import runtime
from ..runtime import torch_device_fn
from .code_cache import config_cache_dir

DEVICE_COUNT = runtime.device.device_count
ATTRS = {
    (2, 2): 5,
    (2, 3): 5,
    (3, 0): 4,
    (3, 1): 4,
    (3, 2): 8,
    (3, 3): 8,
}
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


STRATEGY = {
    None: lambda v: v,
    "log": lambda v: math.ceil(math.log2(v)),
}

TRIU_AUTOTUNE_RECORD_PATH = Path(
    "/workspace/triton_race/auto_config/triu_config_record.md"
)
TRIU_AUTOTUNE_RECORD_NAMES = {"triu_kernel", "triu_batch_kernel"}
TRIU_EFFECTIVE_CONFIG_KEYS = {
    "triu_kernel": ("M_BLOCK_SIZE", "N_BLOCK_SIZE"),
    "triu_batch_kernel": ("BATCH_BLOCK_SIZE", "MN_BLOCK_SIZE"),
}


def _synchronize_for_bench():
    try:
        import torch_tpu

        if hasattr(torch_tpu, "synchronize"):
            torch_tpu.synchronize()
            return
        if hasattr(torch_tpu, "tpu") and hasattr(torch_tpu.tpu, "synchronize"):
            torch_tpu.tpu.synchronize()
            return
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _do_bench(kernel_call, warmup, rep):
    for _ in range(warmup):
        kernel_call()
    _synchronize_for_bench()

    timings = []
    for _ in range(rep):
        _synchronize_for_bench()
        start = time.perf_counter()
        kernel_call()
        _synchronize_for_bench()
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return timings[len(timings) // 2]


def _get_env_positive_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        warnings.warn(f"Ignore invalid {name}={value!r}; use {default}")
        return default
    if parsed <= 0:
        warnings.warn(f"Ignore non-positive {name}={value!r}; use {default}")
        return default
    return parsed


def _format_timing_ms(timing):
    if isinstance(timing, (list, tuple)):
        timing = timing[0]
    if timing == float("inf"):
        return "inf"
    return f"{float(timing):.6f}"


def _config_kwargs(config):
    if hasattr(config, "all_kwargs"):
        return config.all_kwargs()

    return {
        **config.kwargs,
        **{
            k: getattr(config, k)
            for k in ("num_warps", "num_ctas", "num_stages")
            if hasattr(config, k)
        },
    }


def _triu_effective_config_kwargs(fn_name, config):
    kwargs = _config_kwargs(config)
    return {
        key: kwargs[key] for key in TRIU_EFFECTIVE_CONFIG_KEYS[fn_name] if key in kwargs
    }


def _format_best_config(fn_name, config):
    if fn_name in TRIU_AUTOTUNE_RECORD_NAMES:
        return _triu_effective_config_kwargs(fn_name, config)
    return config


def _append_triu_autotune_record(
    fn_name, key, key_names, warmup, rep, timings, best_config, bench_time
):
    if fn_name not in TRIU_AUTOTUNE_RECORD_NAMES:
        return

    TRIU_AUTOTUNE_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for config in timings:
        for column in _triu_effective_config_kwargs(fn_name, config):
            if column not in columns:
                columns.append(column)

    lines = [
        f"## {datetime.now().astimezone().isoformat(timespec='seconds')} `{fn_name}`",
        "",
        f"- tune key names: `{list(key_names)}`",
        f"- tune key: `{key}`",
        f"- warmup: `{warmup}`",
        f"- rep: `{rep}`",
        f"- total bench time: `{bench_time:.6f}s`",
        f"- best config: `{_format_best_config(fn_name, best_config)}`",
        "",
    ]
    if columns:
        lines.append("| idx | " + " | ".join(columns) + " | time_ms | best |")
        lines.append("| --- | " + " | ".join(["---"] * len(columns)) + " | --- | --- |")
        for idx, (config, timing) in enumerate(timings.items(), start=1):
            kwargs = _triu_effective_config_kwargs(fn_name, config)
            values = [str(kwargs.get(column, "")) for column in columns]
            best = "yes" if config is best_config else ""
            lines.append(
                f"| {idx} | "
                + " | ".join(values)
                + f" | {_format_timing_ms(timing)} | {best} |"
            )
    else:
        lines.append("_No config timing data recorded._")
    lines.append("")

    with TRIU_AUTOTUNE_RECORD_PATH.open("a", encoding="utf-8") as record_file:
        record_file.write("\n".join(lines))


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
        share=None,
    ):
        warmup = _get_env_positive_int(
            "FLAGGEMS_AUTOTUNE_WARMUP", 25 if warmup is None else warmup
        )
        rep = _get_env_positive_int(
            "FLAGGEMS_AUTOTUNE_REP", 100 if rep is None else rep
        )

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
        self.strategy = strategy
        self.share = share
        self.do_bench = do_bench
        self.cache = libcache[share] if share else libcache[self.__name__]
        if strategy:
            assert len(self.strategy) == len(self.keys), "Invalid number of strategies"

    def get_key(self, args):
        if self.strategy is None:
            key = [args[k] for k in self.keys if k in args]
            return key
        key = []
        for i, k in enumerate(self.keys):
            s = STRATEGY[self.strategy[i]]
            v = s(args[k])
            key.append(v)
        return key

    def _bench(self, *args, config, **meta):
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources

        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )

        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            try:
                self.fn.run(*args, **current)
            except Exception as e:
                try:
                    self.post_hook(args, exception=e)
                finally:
                    raise
            self.post_hook(args, exception=None)

        try:
            if self.do_bench is not None:
                return self.do_bench(
                    kernel_call, warmup=self.num_warmups, rep=self.num_reps
                )
            return _do_bench(kernel_call, self.num_warmups, self.num_reps)
        except (OutOfResources, CompileTimeAssertionFailure):
            return float("inf")
        except RuntimeError as e:
            if "ppl-compile" in str(e) or "OutOfResources" in str(e):
                return float("inf")
            raise

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            # key = [_args[key] for key in self.keys if key in _args]
            key = self.get_key(_args)
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in pruned_configs
                }
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                full_nargs = {
                    **self.nargs,
                    **kwargs,
                    **self.cache[key].all_kwargs(),
                }
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
                _append_triu_autotune_record(
                    self.base_fn.__name__,
                    key,
                    self.keys,
                    self.num_warmups,
                    self.num_reps,
                    timings,
                    self.cache[key],
                    self.bench_time,
                )
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: "
                f"{_format_best_config(self.base_fn.__name__, self.best_config)};"
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


def libtuner(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
    strategy=None,
    share=None,
):
    """
    Decorator for triton library autotuner.
    """

    def decorator(fn):
        return LibTuner(
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
            share=share,
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
            for key in list(self.signature.parameters.keys()):
                if key in k_args:
                    all_args.append(k_args[key])
                elif key in tune_constexprs:
                    all_args.append(tune_constexprs[key])
                elif key in heur_constexprs:
                    all_args.append(heur_constexprs[key])
            kernel[grid[0:3]](*all_args)
        else:
            kernel[grid[0:3]](*k_args.values())
        return kernel, constexprs


def libentry():
    """
    Decorator for triton library entries.
    """

    def decorator(fn):
        return LibEntry(fn)

    return decorator
