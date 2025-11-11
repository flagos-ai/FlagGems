from __future__ import annotations

import hashlib
import inspect
import logging
import math
import multiprocessing
import os
import sqlite3
import threading
import time
from abc import abstractmethod
from collections import OrderedDict
from functools import cached_property
from itertools import starmap
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
import triton

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend import vendor_module
from flag_gems.utils.code_cache import config_cache_dir

logger = logging.getLogger(__name__)

DEVICE_COUNT = runtime.device.device_count

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

FLAGGEMS_ENABLE_DISK_CACHE = os.getenv("FLAGGEMS_ENABLE_DISK_CACHE", "1") == "1"


class Connections(object):
    def __init__(self, cache_path: str, *args, **kwargs) -> Connections:
        super().__init__(*args, **kwargs)
        self.cache_path: str = cache_path
        self.lock = multiprocessing.Lock()
        self.conns: Dict[int, sqlite3.Connection] = {}

    def __del__(self):
        for conn in self.conns.values():
            conn.close()

    @property
    def curr_conn(self) -> sqlite3.Connection:
        with self.lock:
            tid: int = threading.get_ident()
            conn: Optional[sqlite3.Connection] = self.conns.get(tid)
            if conn is None:
                conn = sqlite3.connect(self.cache_path)
                self.conns[tid] = conn
            return conn


class Cache(object):
    def __init__(self, table_name: str, conns: Connections, *args, **kwargs) -> Cache:
        super().__init__(*args, **kwargs)
        self.table_name: str = table_name
        self.conns: Connections = conns
        self.py2sql: Dict[type, str] = {
            bool: "BOOLEAN",
            int: "INTEGER",
            float: "DOUBLE",
            str: "VARCHAR(16)",  # it often stores string values like 'torch.float16', so 16 would be long enough
        }

    @staticmethod
    def build_key_dict(key: Tuple[Union[int, float, str], ...]) -> Dict[str, Any]:
        return {
            f"key_{k}": v for k, v in enumerate(key) if isinstance(v, (int, float, str))
        }

    @staticmethod
    def build_config_dict(config: triton.Config) -> Dict[str, Any]:
        return {
            k: v
            for k, v in config.all_kwargs().items()
            if isinstance(v, (int, float, str))
        }

    @property
    def conn(self) -> sqlite3.Connection:
        return self.conns.curr_conn


class ConfigCache(Cache):
    """
    `ConfigCache` is used to store the relationship between keys and their known best configurations.
    """

    def __init__(
        self, table_name: str, conns: Connections, *args, **kwargs
    ) -> ConfigCache:
        super().__init__(table_name, conns, *args, **kwargs)
        self.config_signature: inspect.Signature = inspect.signature(triton.Config)
        self.dict_cache: Dict[
            Tuple[Union[int, float, str], ...], triton.Config
        ] = {}  # this dict is used to cache some results in the memory
        self.lock = multiprocessing.Lock()
        self.names: List[str] = [
            name
            for _, name, _, _, _, _ in self.conn.execute(
                f"PRAGMA table_info({self.table_name});"
            ).fetchall()
        ]
        self.create_sql: Optional[
            str
        ] = None  # if the corresponding sql instruction is None, meaning the table is not ready, we need to flush it
        self.insert_sql: Optional[str] = None

    def __contains__(self, key: Tuple[Union[int, float, str], ...]) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: Tuple[Union[int, float, str], ...]) -> triton.Config:
        ret: Optional[triton.Config] = self.get(key)
        if ret is None:
            raise KeyError(key)
        else:
            return ret

    def __setitem__(
        self, key: Tuple[Union[int, float, str], ...], config: triton.Config
    ):
        self.set(key, config)

    @property
    def knames(self) -> Iterator[str]:
        return filter(
            lambda name: name.startswith("key_"), self.names
        )  # if the column name starts with "key_", it should be a key defined by the libtuner

    @property
    def cnames(self) -> Iterator[str]:
        return filter(
            lambda name: not name.startswith("key_"), self.names
        )  # otherwise, it should be a parameter in the config

    @property
    def select_sql(self) -> Optional[str]:
        if self.names:
            return "SELECT {} FROM {} WHERE {};".format(
                ", ".join(self.cnames),
                self.table_name,
                " AND ".join(map(lambda kname: f"{kname}=?", self.knames)),
            )
        else:
            return None

    def get(self, key: Tuple[Union[int, float, str], ...]) -> Optional[triton.Config]:
        with self.lock:
            ret = self.dict_cache.get(key)
            if ret is not None or self.select_sql is None:
                # if the key is already in the dict cache, we can return it directly
                # or if `select_sql` is not ready yet, which means the table is not ready yet, we can return None
                return ret
            rets = self.conn.execute(self.select_sql, key).fetchall()
            if not rets:
                return None
            [ret] = rets
            kwargs: Dict[str, Any] = {}
            numargs: Dict[str, int] = {}
            for k, v in zip(self.cnames, ret):
                if k in self.config_signature.parameters:
                    numargs[k] = v
                else:
                    kwargs[k] = v
            ret = triton.Config(kwargs, **numargs)
            self.dict_cache[key] = ret
            return ret

    def set(self, key: Tuple[Union[int, float, str], ...], config: triton.Config):
        with self.lock:
            key_queries: Dict[str, Any] = self.build_key_dict(key)
            config_queries: Dict[str, Any] = self.build_config_dict(config)
            queries: Dict[str, Any] = key_queries | config_queries
            if self.create_sql is None:
                self.create_sql = "CREATE TABLE IF NOT EXISTS {} ({});".format(
                    self.table_name,
                    ",\n".join(
                        [f"{k} {self.py2sql[type(v)]}" for k, v in queries.items()]
                        + ["PRIMARY KEY ({})".format(", ".join(key_queries.keys()))]
                    ),
                )
                self.conn.execute(self.create_sql)
            if not self.names:
                self.names = queries
            if self.insert_sql is None:
                self.insert_sql = "INSERT OR REPLACE INTO {} VALUES ({});".format(
                    self.table_name, ", ".join("?" for _ in queries.values())
                )
            self.conn.execute(self.insert_sql, [*queries.values()])
        self.conn.commit()


class BenchmarkCache(Cache):
    def __init__(
        self,
        table_name: str,
        key: Tuple[Union[int, float, str], ...],
        conns: Connections,
        *args,
        **kwargs,
    ) -> BenchmarkCache:
        """
        `BenchmarkCache` is used to store the benchmark results for the pair of the specific key and configuration.
        """
        super().__init__(table_name, conns, *args, **kwargs)
        self.key: Tuple[Union[int, float, str], ...] = key
        self.lock = multiprocessing.Lock()
        self.create_sql: Optional[str] = None
        self.select_sql: Optional[str] = None
        self.insert_sql: Optional[str] = None

    def __getitem__(self, config: triton.Config) -> Optional[List[float]]:
        with self.lock:
            queries: Dict[str, Union[int, float, str]] = self.build_query(config)
            if self.select_sql is None:
                where: str = " AND ".join(f"{k} = ?" for k in queries.keys())
                self.select_sql = (
                    f"SELECT p50, p20, p80 FROM {self.table_name} WHERE {where};"
                )
            ret = self.conn.execute(self.select_sql, [*queries.values()]).fetchone()
            if isinstance(ret, tuple):
                ret = [*ret]
            return ret

    def __setitem__(self, config: triton.Config, benchmark: List[float]) -> None:
        with self.lock:
            queries: Dict[str, Union[int, float, str]] = self.build_query(config)
            values: List[str] = [*queries.values(), *benchmark]
            if not self.insert_sql:
                self.insert_sql = "INSERT OR REPLACE INTO {} VALUES ({});".format(
                    self.table_name, ", ".join("?" for _ in values)
                )
            self.conn.execute(self.insert_sql, values)
        self.conn.commit()

    def build_query(self, config: triton.Config) -> Dict[str, Any]:
        queries: Dict[str, Any] = self.build_key_dict(
            self.key
        ) | self.build_config_dict(config)
        if self.create_sql is None:
            self.create_sql = "CREATE TABLE IF NOT EXISTS {} ({});".format(
                self.table_name,
                ",\n".join(
                    [f"{k} {self.py2sql[type(v)]}" for k, v in queries.items()]
                    + [f"p{n} DOUBLE" for n in [50, 20, 80]]
                    + ["PRIMARY KEY ({})".format(", ".join(queries.keys()))]
                ),
            )
            self.conn.execute(self.create_sql)
            self.conn.commit()
        return queries


class LibCache(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LibCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, enable_disk_cache: bool):
        self.global_cache: Dict = {}
        self.volumn: Dict = {}
        cache_file_name = (
            f"TunedConfig_{torch.cuda.get_device_name().replace(' ', '_')}_triton_{major_version}_{minor_version}.db"
            if vendor_module.vendor_info.vendor_name == "nvidia"
            else f"TunedConfig_{vendor_module.vendor_info.vendor_name}_triton_{major_version}_{minor_version}.db"
        )
        self.cache_path = (
            (config_cache_dir() / cache_file_name) if enable_disk_cache else ":memory:"
        )
        self.bench_lock = multiprocessing.Lock()
        self.config_lock = multiprocessing.Lock()
        self.conns: Connections = Connections(self.cache_path)
        self.config_cache_pool: Dict[str, ConfigCache] = {}
        self.benchmark_cache_pool: Dict[
            Tuple[str, Tuple[Union[int, float, str], ...]], BenchmarkCache
        ] = {}

    def __post_init__(self):
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("PRAGMA journal=WAL;")
            conn.commit()

    def __getitem__(
        self, key: Union[str, Tuple[Union[int, float, str], ...]]
    ) -> Union[BenchmarkCache, ConfigCache]:
        if isinstance(key, str):
            return self.get_config(key)
        elif isinstance(key, tuple):
            return self.get_benchmark(*key)
        else:
            assert False, f"the type of key '{key.__class__.__name__}' is unacceptable"

    def get_benchmark(
        self, table: str, key: Tuple[Union[int, float, str], ...]
    ) -> BenchmarkCache:
        with self.bench_lock:
            ret = self.benchmark_cache_pool.get((table, key))
            if ret is None:
                ret = BenchmarkCache(table, key, self.conns)
                self.benchmark_cache_pool[(table, key)] = ret
        return ret

    def get_config(self, table: str) -> ConfigCache:
        with self.config_lock:
            ret = self.config_cache_pool.get(table)
            if ret is None:
                ret = ConfigCache(table, self.conns)
                self.config_cache_pool[table] = ret
        return ret


libcache = LibCache(FLAGGEMS_ENABLE_DISK_CACHE)


class LibTuner(triton.runtime.Autotuner):
    """`LibTuner` is the base class for `FlagGems` library autotuner.

    It could be extended in two ways, overriding the `policy` or `run` method in a subclass.
    For `policy` extension, `LibTuner` provides a decorator `register_policy` to register a policy function quickly.
    Please refer to the implementation of `default_policy` for an example.
    """

    # The dispatch table for `LibTuner` subclasses. It's shared across all instances.
    _dispatch_table: Dict[str, Type[LibTuner]] = {}
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
        self.strategy: List[Callable[[Any], Any]] = strategy
        self.config_table_name: str = f"{self.__name__}_{self.kernel_hash}"
        self.benchmark_table_name: str = f"{self.__name__}_{self.cache_key}_benchmark"
        self.cache: BenchmarkCache = libcache[self.config_table_name]

    @cached_property
    def cache_key(self):
        jit_fn = self.fn
        while not isinstance(jit_fn, triton.runtime.JITFunction):
            jit_fn = jit_fn.fn
        return jit_fn.cache_key

    @cached_property
    def kernel_hash(self):
        return hashlib.md5(
            f"{self.cache_key}{self.configs_hash}".encode("utf-8")
        ).hexdigest()

    @cached_property
    def configs_hash(self):
        return hashlib.md5(
            ",".join(map(lambda config: str(config), self.configs)).encode("utf-8")
        ).hexdigest()

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
        fn: Callable[[triton.Config], List[float]],
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
    ) -> Type[LibTuner]:
        """A decorator to register a policy for `LibTuner`.

        This decorator allows you to create a new `LibTuner` subclass without defining a new class explicitly.
        The new subclass will have the `policy` method set to the provided policy function and will be registered under
        the specified name in the `LibTuner` dispatch table.
        """

        def decorator(
            policy_impl: Callable[
                [
                    Callable[[triton.Config], List[float]],
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
                    fn: Callable[[triton.Config], List[float]],
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
                cache: BenchmarkCache = libcache[self.benchmark_table_name, key]
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()

                def bench(config: triton.Config) -> List[float]:
                    ret = cache[config]
                    if ret is None:
                        ret = self._bench(*args, config=config, **kwargs)
                        cache[config] = ret
                    return ret

                best_config, timings = self.policy(
                    bench,
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
                f"{self.bench_time:.2f}s; key info: {key}, best config selected: {self.best_config};"
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
    return 2 ** math.ceil(math.log2(key))


@LibTuner.register_strategy("align32")
def align32_strategy(key: Union[int, float]) -> int:
    return math.ceil(key / 32) * 32


@LibTuner.register_policy("default")
def default_policy(
    bench_fn: Callable[[triton.Config], List[float]],
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
            bench_fn: Callable[[triton.Config], List[float]],
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
        self.lock = multiprocessing.Lock()
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
                if major_version == 3 and 3 <= minor_version <= 5:
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
                if major_version == 3 and 3 <= minor_version <= 5:
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

        if major_version == 3 and 3 <= minor_version <= 5:
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
