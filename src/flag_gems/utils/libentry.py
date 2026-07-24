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

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import math
import multiprocessing
import os
import time
from abc import abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from enum import Enum
from functools import cached_property
from itertools import starmap
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import triton

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.runtime.backend import _state
from flag_gems.utils.code_cache import config_cache_dir
from flag_gems.utils.models import PersistantModel, SQLPersistantModel

logger = logging.getLogger(__name__)

DEVICE_COUNT = runtime.device.device_count

version = triton.__version__.split(".")
major_version, minor_version = eval(version[0]), eval(version[1])


class LibTunerRunMode(str, Enum):
    """Control one scoped :class:`LibTuner.run` config-selection pass."""

    NORMAL = "normal"
    FORCE_POLICY = "force_policy"
    EXHAUSTIVE_COLLECTION = "exhaustive_collection"


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

FLAGGEMS_DB_URL = os.getenv("FLAGGEMS_DB_URL", None)
BENCHMARK_CACHE_SCHEMA_VERSION = 2
DEFAULT_BENCHMARK_WARMUP_MS = 25
DEFAULT_BENCHMARK_REP_MS = 100


def _infer_tensor_dtypes(values: Iterable[Any]) -> Tuple[Any, ...]:
    """Return dtypes of tensor kernel arguments in their argument order.

    Only actual ``torch.Tensor`` instances and Triton ``TensorDescriptor``
    instances backed by a tensor contribute a dtype.  This deliberately avoids
    treating an arbitrary object with a ``dtype`` attribute as a tensor.  TMA
    descriptors expose their dtype through ``descriptor.base.dtype``.  The
    result is shared by FlagTune identity, LibEntry dispatch keys, and
    LibTuner's persistent cache keys.
    """
    try:
        import torch
    except ImportError:
        return ()

    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except ImportError:
        TensorDescriptor = ()

    dtypes = []
    for value in values:
        if isinstance(value, torch.Tensor):
            dtypes.append(value.dtype)
        elif isinstance(value, TensorDescriptor) and isinstance(
            value.base, torch.Tensor
        ):
            dtypes.append(value.base.dtype)
    return tuple(dtypes)


class Cache(object):
    def __init__(
        self, table_name: str, model: PersistantModel, *args, **kwargs
    ) -> Cache:
        super().__init__(*args, **kwargs)
        self.table_name: Final[str] = table_name
        self.model: Final[PersistantModel] = model


class ConfigCache(Cache):
    """
    `ConfigCache` is used to store the relationship between keys and their known best configurations.
    """

    def __init__(
        self, table_name: str, model: PersistantModel, *args, **kwargs
    ) -> ConfigCache:
        super().__init__(table_name, model, *args, **kwargs)

    def __contains__(self, key: Tuple[Union[int, float, str], ...]) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: Tuple[Union[int, float, str], ...]) -> triton.Config:
        ret: Optional[triton.Config] = self.get(key)
        if ret is None:
            raise KeyError(f"Key {key} not found in ConfigCache.")
        return ret

    def __setitem__(
        self, key: Tuple[Union[int, float, str], ...], config: triton.Config
    ) -> None:
        self.set(key, config)

    def get(self, key: Tuple[Union[int, float, str], ...]) -> Optional[triton.Config]:
        return self.model.get_config(self.table_name, key)

    def set(
        self, key: Tuple[Union[int, float, str], ...], config: triton.Config
    ) -> None:
        return self.model.put_config(self.table_name, key, config)


class BenchmarkCache(Cache):
    def __init__(
        self,
        table_name: str,
        model: PersistantModel,
        key: Tuple[Union[int, float, str], ...],
        *args,
        **kwargs,
    ) -> BenchmarkCache:
        """
        `BenchmarkCache` is used to store the benchmark results for the pair of the specific key and configuration.
        """
        super().__init__(table_name, model, *args, **kwargs)
        self.key: Final[Tuple[Union[int, float, str], ...]] = key

    def __contains__(self, config: triton.Config) -> bool:
        return self.model.get_benchmark(self.key, config) is not None

    def __getitem__(self, config: triton.Config) -> Tuple[float]:
        ret: Optional[Tuple[float, float, float]] = self.get(config)
        if ret is None:
            raise KeyError(
                f"Config {config} not found in BenchmarkCache for key {self.key}."
            )
        return ret

    def __setitem__(self, config: triton.Config, benchmark: Tuple[float]) -> None:
        return self.set(config, benchmark)

    def get(self, config: triton.Config) -> Optional[Tuple[float, float, float]]:
        return self.model.get_benchmark(self.table_name, self.key, config)

    def set(self, config: triton.Config, benchmark: Tuple[float, float, float]) -> None:
        return self.model.put_benchmark(self.table_name, self.key, config, benchmark)


class LibCache(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LibCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_url: Optional[str] = None):
        self.global_cache: Dict = {}
        self.volumn: Dict = {}
        vendor_name = _state.vendor_module.vendor_info.vendor_name
        if db_url is None:
            cache_file_name: str = (
                f"TunedConfig_{vendor_name}_triton_{major_version}_{minor_version}.db"
            )
            cache_path: Path = config_cache_dir() / cache_file_name
            self.db_url: str = f"sqlite:///{cache_path}"
        else:
            self.db_url: str = db_url
        self.config_cache_pool: Dict[str, ConfigCache] = {}
        self.benchmark_cache_pool: Dict[
            Tuple[str, Tuple[Union[int, float, str], ...]], BenchmarkCache
        ] = {}
        self.model: PersistantModel = SQLPersistantModel(self.db_url)

    @overload
    def __getitem__(self, key: str) -> ConfigCache: ...

    @overload
    def __getitem__(self, key: Tuple[Union[int, float, str]]) -> BenchmarkCache: ...

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
        ret = self.benchmark_cache_pool.get((table, key))
        if ret is None:
            ret = BenchmarkCache(table, self.model, key)
            self.benchmark_cache_pool[(table, key)] = ret
        return ret

    def get_config(self, table: str) -> ConfigCache:
        ret = self.config_cache_pool.get(table)
        if ret is None:
            ret = ConfigCache(table, self.model)
            self.config_cache_pool[table] = ret
        return ret


libcache = LibCache(FLAGGEMS_DB_URL)


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
        flagtune_op_name=None,
        flagtune_expand_op_name=None,
        flagtune_op_id=None,
        flagtune_pre_hook=None,
        # Args from new FlagTune
        flagtune_variant=None,
        flagtune_yaml_path=None,
        flagtune_dtype_resolver=None,
    ):
        """Initialize library autotuning and optional FlagTune integration.

        The standard Triton arguments configure baseline tuning, caching, and
        benchmarking.  The FlagTune-specific arguments are:

        Args:
            flagtune_op_name: FlagGems legacy expanded-config enablement key.
                It never participates in FlagTree bundle identity. ``None``
                disables legacy config-space switching for this tuner.
            flagtune_expand_op_name: Optional FlagGems name used to load the
                legacy expanded config space.  It defaults to
                ``flagtune_op_name`` and is not a FlagTune variant name.
            flagtune_op_id: Globally namespaced FlagTree operator identity.
            flagtune_variant: Single-segment implementation/model variant.
            flagtune_yaml_path: Optional FlagGems expanded-config YAML path used
                by :meth:`apply_flagtune`; this is separate from the new
                operator-registration YAML.
            flagtune_pre_hook: Hook attached to fresh Triton configs produced by
                the proposer when they do not already provide one.
            flagtune_dtype_resolver: Optional trusted Python callable returning
                tensor dtypes in model identity order. YAML cannot set it.

        Notes:
            ``USE_FLAGTUNE``/FlagGems runtime enablement switches to the legacy
            expanded config space and intentionally bypasses the FlagTree
            proposer policy.  FlagTree's own enablement is
            ``TRITON_USE_FLAGTUNE``.  The similar names represent independent
            mechanisms and must not be treated as aliases.
        """
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
        elif major_version == 3 and minor_version <= 1:
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
        else:
            # Triton 3.2+ removed warmup/rep/use_cuda_graph positional arguments.
            # Preserve FlagGems tuning behavior by translating them into do_bench.
            if do_bench is None:
                if use_cuda_graph:
                    from triton.testing import do_bench_cudagraph

                    def do_bench(kernel_call, quantiles):
                        return do_bench_cudagraph(
                            kernel_call,
                            rep=rep if rep is not None else 100,
                            quantiles=quantiles,
                        )

                elif warmup is not None or rep is not None:

                    def do_bench(kernel_call, quantiles):
                        return triton.testing.do_bench(
                            kernel_call,
                            warmup=warmup if warmup is not None else 25,
                            rep=rep if rep is not None else 100,
                            quantiles=quantiles,
                        )

            super().__init__(
                fn,
                arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                pre_hook=pre_hook,
                post_hook=post_hook,
                prune_configs_by=prune_configs_by,
                do_bench=do_bench,
            )
        if use_cuda_graph:
            self._benchmark_protocol = (
                "triton_do_bench_cudagraph",
                0,
                rep if rep is not None else DEFAULT_BENCHMARK_REP_MS,
            )
        elif do_bench is not None and warmup is None and rep is None:
            self._benchmark_protocol = ("custom_do_bench", -1, -1)
        else:
            self._benchmark_protocol = (
                "triton_do_bench",
                warmup if warmup is not None else DEFAULT_BENCHMARK_WARMUP_MS,
                rep if rep is not None else DEFAULT_BENCHMARK_REP_MS,
            )
        self.__name__ = self.base_fn.__name__
        self.keys = key
        self.strategy: List[Callable[[Any], Any]] = self._normalize_strategy(strategy)
        self.config_table_name: str = f"{self.__name__}_{self.kernel_hash}"
        self.benchmark_table_name: str = (
            f"{self.__name__}_{self.cache_key}_benchmark_v"
            f"{BENCHMARK_CACHE_SCHEMA_VERSION}"
        )
        self.cache: ConfigCache = libcache[self.config_table_name]
        self._flagtune_default_configs = self.configs
        self._flagtune_default_strategy = strategy
        self._flagtune_active = False
        self._flagtune_warned = False
        self._flagtune_op_name = flagtune_op_name
        self._flagtune_expand_op_name = flagtune_expand_op_name or flagtune_op_name
        if (flagtune_op_id is None) != (flagtune_variant is None):
            raise ValueError(
                "flagtune_op_id and flagtune_variant must be supplied together"
            )
        self._flagtune_op_id = flagtune_op_id
        self._flagtune_variant = flagtune_variant
        self._flagtune_yaml_path = flagtune_yaml_path
        self._flagtune_pre_hook = flagtune_pre_hook
        if flagtune_dtype_resolver is not None and not callable(
            flagtune_dtype_resolver
        ):
            raise TypeError("flagtune_dtype_resolver must be a trusted callable")
        self._flagtune_dtype_resolver = flagtune_dtype_resolver
        self._run_mode = LibTunerRunMode.NORMAL

    def _normalize_strategy(self, strategy):
        if isinstance(strategy, str):
            strategy = LibTuner.get_strategy(strategy)
        if not isinstance(strategy, (list, tuple)):
            strategy = [strategy] * len(self.keys)
        assert len(strategy) == len(
            self.keys
        ), f"the length of strategy {len(strategy)} must match the length of keys {len(self.keys)}"
        return [LibTuner.get_strategy(s) if isinstance(s, str) else s for s in strategy]

    def _set_configs_and_strategy(self, configs, strategy):
        self.configs = configs
        self.strategy = self._normalize_strategy(strategy)
        self.__dict__.pop("configs_hash", None)
        self.__dict__.pop("kernel_hash", None)
        self.config_table_name = f"{self.__name__}_{self.kernel_hash}"
        self.benchmark_table_name = (
            f"{self.__name__}_{self.cache_key}_benchmark_v"
            f"{BENCHMARK_CACHE_SCHEMA_VERSION}"
        )
        self.cache = libcache[self.config_table_name]

    def apply_flagtune(self):
        if self._flagtune_op_name is None:
            return False

        enabled = runtime.flagtune_enabled(self._flagtune_op_name)
        if enabled == self._flagtune_active:
            return False

        if not enabled:
            self._set_configs_and_strategy(
                self._flagtune_default_configs,
                self._flagtune_default_strategy,
            )
            self._flagtune_active = False
            return True

        expand_config = runtime.get_expand_config(
            self._flagtune_expand_op_name,
            yaml_path=self._flagtune_yaml_path,
        )
        configs = runtime.ops_get_configs(
            self._flagtune_expand_op_name,
            yaml_path=self._flagtune_yaml_path,
            pre_hook=self._flagtune_pre_hook,
        )
        if expand_config == -1 or not configs:
            if not self._flagtune_warned:
                logger.warning(
                    "FlagTune expand config is unavailable for %s; using default configs.",
                    self._flagtune_expand_op_name,
                )
                self._flagtune_warned = True
            return False

        self._set_configs_and_strategy(configs, expand_config["strategy"])
        self._flagtune_active = True
        return True

    @cached_property
    def cache_key(self) -> str:
        jit_fn = self.fn
        while not isinstance(jit_fn, triton.runtime.JITFunction):
            jit_fn = jit_fn.fn
        return jit_fn.cache_key

    @cached_property
    def kernel_hash(self) -> str:
        return hashlib.md5(
            f"{self.cache_key}{self.configs_hash}".encode("utf-8")
        ).hexdigest()[:32]

    @cached_property
    def configs_hash(self) -> str:
        return hashlib.md5(
            ",".join(map(lambda config: str(config), self.configs)).encode("utf-8")
        ).hexdigest()[:32]

    def get_key(self, args):
        """Return the strategy-normalized key used only by ConfigCache."""
        if self.strategy is None:
            key = tuple(args[k] for k in self.keys if k in args)
        else:
            key = tuple(
                starmap(
                    lambda idx0, idx1: self.strategy[idx0](args[idx1]),
                    enumerate(self.keys),
                )
            )
        key += tuple(str(dtype) for dtype in _infer_tensor_dtypes(args.values()))
        return key

    def get_benchmark_key(self, args):
        """Return an exact shape and measurement-protocol BenchmarkCache key.

        ConfigCache deliberately applies strategies such as ``align32`` so
        nearby shapes can share a known best config. BenchmarkCache stores
        training labels and must instead preserve every raw autotune key.
        Appending the active benchmark protocol prevents measurements collected
        with different warmup/repetition durations from sharing one row.
        """
        key = tuple(args[k] for k in self.keys if k in args)
        key += tuple(str(dtype) for dtype in _infer_tensor_dtypes(args.values()))
        return key + tuple(self._benchmark_protocol)

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
                    return policy_impl(self, fn, configs, args, kwargs)

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

    @contextmanager
    def use_run_mode(self, mode: Union[LibTunerRunMode, str]) -> Iterator[None]:
        """Apply an explicit config-selection mode to one scoped operator call.

        Args:
            mode: ``normal`` uses ConfigCache normally; ``force_policy`` skips
                ConfigCache while preserving the active policy; and
                ``exhaustive_collection`` skips ConfigCache and forces the
                exhaustive default policy while retaining BenchmarkCache.

        Yields:
            Control to exactly one or more calls made by the caller while the
            selected mode is active.

        Raises:
            ValueError: If ``mode`` is not a supported
                :class:`LibTunerRunMode`.

        Implementation:
            The previous instance-local mode is restored in ``finally``. This
            keeps offline benchmark behavior explicit and prevents process-wide
            environment state from leaking into normal runtime calls.
        """
        selected = LibTunerRunMode(mode)
        previous = getattr(self, "_run_mode", LibTunerRunMode.NORMAL)
        self._run_mode = selected
        try:
            yield
        finally:
            self._run_mode = previous

    @contextmanager
    def use_benchmark_protocol(self, warmup: int, rep: int) -> Iterator[None]:
        """Scope the exact Triton timing protocol encoded by BenchmarkCache.

        Args:
            warmup: Non-negative Triton warmup duration in milliseconds.
            rep: Positive Triton measurement duration in milliseconds.

        The previous protocol is restored on exit so offline Train/Pretune
        overrides cannot leak into normal runtime autotuning.
        """
        if warmup < 0:
            raise ValueError("benchmark warmup must be non-negative")
        if rep <= 0:
            raise ValueError("benchmark repetition duration must be positive")
        previous = self._benchmark_protocol
        self._benchmark_protocol = ("triton_do_bench", int(warmup), int(rep))
        try:
            yield
        finally:
            self._benchmark_protocol = previous

    def benchmark_config(
        self,
        config: triton.Config,
        *,
        warmup: int,
        rep: int,
        quantiles: Tuple[float, ...] = (0.5, 0.2, 0.8),
        args: Optional[Tuple[Any, ...]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Freshly benchmark one fixed config through the LibTuner kernel path.

        Args:
            config: Exact Triton config to execute; no policy or cache lookup is
                performed.
            warmup: Triton benchmark warmup duration in milliseconds.
            rep: Triton benchmark measurement duration in milliseconds.
            quantiles: Quantiles returned in caller-specified order.
            args: Optional low-level kernel arguments. When omitted, reuse the
                most recent arguments captured by :meth:`run`.
            meta: Optional low-level kernel keyword arguments. When omitted,
                reuse the most recent metadata captured by :meth:`run`.

        Returns:
            Fresh timing samples in milliseconds and in ``quantiles`` order.

        Raises:
            ValueError: If durations or quantiles are invalid.
            RuntimeError: If no prior kernel context is available and explicit
                ``args``/``meta`` were not provided.

        Implementation:
            The method temporarily installs an explicit Triton ``do_bench``
            configuration and delegates to Autotuner ``_bench``. It deliberately
            bypasses both ConfigCache and BenchmarkCache. Resetting
            ``seen_tuned_metas`` prevents AABS deduplication from returning an
            earlier sample instead of executing the requested fresh trial.

        Limitations:
            This measures fixed-config kernel device time. Public operator
            allocation, Python dispatch, and model/config selection are outside
            the timed region.
        """
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if rep <= 0:
            raise ValueError("rep must be positive")
        if not quantiles or any(value < 0 or value > 1 for value in quantiles):
            raise ValueError("quantiles must contain values between 0 and 1")
        benchmark_quantiles = tuple(quantiles)
        benchmark_args = (
            tuple(args)
            if args is not None
            else getattr(self, "_last_benchmark_args", None)
        )
        benchmark_meta = (
            dict(meta)
            if meta is not None
            else getattr(self, "_last_benchmark_meta", None)
        )
        if benchmark_args is None or benchmark_meta is None:
            raise RuntimeError(
                "benchmark_config requires explicit args/meta or a prior LibTuner.run"
            )

        original_do_bench = self.do_bench
        original_nargs = getattr(self, "nargs", None)

        def configured_do_bench(kernel_call, quantiles):
            return triton.testing.do_bench(
                kernel_call,
                warmup=warmup,
                rep=rep,
                quantiles=benchmark_quantiles,
            )

        self.do_bench = configured_do_bench
        self.nargs = dict(zip(self.arg_names, benchmark_args))
        if hasattr(self, "seen_tuned_metas"):
            self.seen_tuned_metas = {}
        try:
            return list(
                self._bench(
                    *benchmark_args,
                    config=config,
                    **benchmark_meta,
                )
            )
        finally:
            self.do_bench = original_do_bench
            self.nargs = original_nargs

    def run(self, *args, **kwargs):
        """Select and launch a config under the current scoped run mode.

        Normal execution queries and updates the persistent shape-to-best-config
        :class:`ConfigCache`. :meth:`use_run_mode` can instead force the active
        policy without touching ConfigCache, or force exhaustive default-policy
        collection while still reading and filling the per-config
        :class:`BenchmarkCache`.

        ``benchmark_success_count`` counts finite cache misses measured by this
        call. ``benchmark_cache_hit_count`` counts reused latency entries. Both
        counters reset for every invocation. BenchmarkCache v2 uses the exact
        raw autotune key plus tensor dtypes and the scoped warmup/repetition
        protocol; ConfigCache continues to use the strategy-normalized key.
        """
        self.benchmark_success_count = 0
        self.benchmark_cache_hit_count = 0
        run_mode = LibTunerRunMode(
            getattr(self, "_run_mode", LibTunerRunMode.NORMAL)
        )
        bypass_config_cache = run_mode is not LibTunerRunMode.NORMAL
        exhaustive_collection = (
            run_mode is LibTunerRunMode.EXHAUSTIVE_COLLECTION
        )
        if hasattr(self, "seen_tuned_metas"):
            self.seen_tuned_metas = {}  # flagtree aabs: deduplicate tuned meta
        self._last_benchmark_args = tuple(args)
        self._last_benchmark_meta = dict(kwargs)
        # `arg_names` corresponds to the arguments of the `JITFunction`'s signature,
        # so please make sure the orders of `arg_names` and `args` match.
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1 or bypass_config_cache:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for k, v in all_args.items() if k in self.arg_names}
            config_key = self.get_key(_args)
            benchmark_key = self.get_benchmark_key(_args)
            if bypass_config_cache or config_key not in self.cache:
                cache: BenchmarkCache = libcache[
                    self.benchmark_table_name, benchmark_key
                ]
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()

                def bench(config: triton.Config) -> List[float]:
                    ret = cache.get(config)
                    if ret is None:
                        ret = self._bench(*args, config=config, **kwargs)
                        if ret and all(math.isfinite(float(value)) for value in ret):
                            self.benchmark_success_count += 1
                        cache[config] = tuple(ret)
                    else:
                        self.benchmark_cache_hit_count += 1
                    return list(ret)

                if exhaustive_collection:
                    best_config, timings = LibTuner.get("default").policy(
                        self,
                        bench,
                        pruned_configs,
                        args,
                        kwargs,
                    )
                else:
                    best_config, timings = self.policy(
                        bench,
                        pruned_configs,
                        args,
                        kwargs,
                    )
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                if not bypass_config_cache:
                    self.cache[config_key] = best_config
                    config = self.cache[config_key]
                else:
                    config = best_config
                full_nargs = {
                    **self.nargs,
                    **kwargs,
                    **config.all_kwargs(),
                }
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
            else:
                config = self.cache[config_key]
            if config.pre_hook is None:
                cached_kwargs = config.all_kwargs()
                for original_config in self.configs:
                    if original_config.all_kwargs() == cached_kwargs:
                        # Use the original config which has the pre_hook
                        config = original_config
                        break
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; config key: {config_key}, "
                f"benchmark key: {benchmark_key}, "
                f"best config selected: {self.best_config};"
            )
        full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
        if (
            hasattr(self, "shared_config_pre_hook")
            and self.shared_config_pre_hook is not None
        ):
            self.shared_config_pre_hook(full_nargs)
        elif config.pre_hook is not None:
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret


# ---- FlagTune Proposer Integration ----

_FLAGTUNE_PROPOSER_POOL: Dict[Any, Any] = {}
_FLAGTUNE_VARIANT_INFO_POOL: Dict[Any, Any] = {}
_FLAGTUNE_AVAILABILITY: Optional[Tuple[bool, Optional[BaseException]]] = None
_FLAGTUNE_UNAVAILABLE_WARNED = False
_FLAGTUNE_FALLBACK_WARNED: set = set()


def _flagtune_available() -> Tuple[bool, Optional[BaseException]]:
    global _FLAGTUNE_AVAILABILITY
    if _FLAGTUNE_AVAILABILITY is None:
        try:
            from triton.flagtune.predict import (  # noqa: F401
                load_model_bundle,
                make_config_proposer,
            )

            _FLAGTUNE_AVAILABILITY = (True, None)
        except Exception as exc:
            _FLAGTUNE_AVAILABILITY = (False, exc)
    return _FLAGTUNE_AVAILABILITY


def _flagtune_enabled() -> bool:
    """Return FlagTree's cached enablement independently of legacy FlagGems."""
    from triton.flagtune import is_enabled

    return is_enabled()


def _warn_flagtune_unavailable_once(exc: Optional[BaseException]) -> None:
    global _FLAGTUNE_UNAVAILABLE_WARNED
    if _FLAGTUNE_UNAVAILABLE_WARNED:
        return
    logger.warning(
        "FlagTune requested but triton.flagtune is unavailable: %s; using default policy.",
        exc,
    )
    _FLAGTUNE_UNAVAILABLE_WARNED = True


def _warn_flagtune_fallback_once(reason: str, message: str, *args) -> None:
    if reason in _FLAGTUNE_FALLBACK_WARNED:
        return
    logger.warning(message, *args)
    _FLAGTUNE_FALLBACK_WARNED.add(reason)


def _ensure_flagtune_proposer(identity):
    """Return cached proposer and bundled metadata for one operator variant.

    Args:
        op_id: Globally namespaced logical operator identifier.
        variant: Single-segment implementation/model variant.

    Returns:
        ``(ConfigProposer, VariantInfo)`` stored in process-global pools keyed by
        the operator/variant tuple.

    Raises:
        FileNotFoundError: If the model bundle cannot be resolved.
        Exception: Model dependency or config errors from proposer creation are
            intentionally propagated to :func:`flagtune_policy`, which logs a
            warning once and falls back.

    Notes:
        Cached objects are never refreshed. Replacing model files later in the
        process does not invalidate these pools.
    """
    if identity not in _FLAGTUNE_PROPOSER_POOL:
        from triton.flagtune.predict import load_model_bundle, make_config_proposer

        _FLAGTUNE_PROPOSER_POOL[identity] = make_config_proposer(
            identity.op_id,
            identity.variant,
            gpu_key=identity.gpu_key,
            dtype_key=identity.dtype_key,
        )
        _FLAGTUNE_VARIANT_INFO_POOL[identity] = load_model_bundle(
            identity.op_id,
            identity.variant,
            gpu_key=identity.gpu_key,
            dtype_key=identity.dtype_key,
        ).variant

    return _FLAGTUNE_PROPOSER_POOL[identity], _FLAGTUNE_VARIANT_INFO_POOL[identity]


def _configs_to_dicts_for_proposer(
    configs: Iterator[triton.Config], param_fields: List[str]
) -> List[Dict[str, Any]]:
    """Convert LibTuner configs to the proposer's parameter dictionaries.

    Args:
        configs: Possibly one-shot iterator of ``triton.Config`` objects.
        param_fields: Exact variant parameter names to copy from ``cfg.kwargs``.

    Returns:
        A materialized list containing declared kernel parameters plus available
        ``num_warps``, ``num_stages``, and ``num_ctas`` launch metadata.  All
        copied values are converted to integers and empty entries are skipped.

    Notes:
        Extra constexpr arguments, pre-hooks, and other Config state are not
        represented.  Missing declared fields are not rejected here; the
        proposer currently accepts this list only for interface compatibility.
    """
    result = []
    for cfg in configs:
        d: Dict[str, Any] = {}
        for f in param_fields:
            if f in getattr(cfg, "kwargs", {}):
                d[f] = int(cfg.kwargs[f])
        for attr in ("num_warps", "num_stages", "num_ctas"):
            if hasattr(cfg, attr):
                d[attr] = int(getattr(cfg, attr))
        if d:
            result.append(d)
    return result


def _make_proposer_bench_adapter(
    bench_fn: Callable[[triton.Config], List[float]],
    to_config: Callable[[Dict[str, Any]], triton.Config],
):
    """Adapt a LibTuner benchmark callable to FlagTune's dictionary contract.

    Args:
        bench_fn: Callable accepting ``triton.Config`` and returning latency
            samples, normally backed by LibTuner's ``BenchmarkCache``.
        to_config: Variant converter from a complete parameter dictionary to a
            fresh ``triton.Config``.

    Returns:
        A ``BenchmarkFn(config_dict, n_runs=None)`` closure.  It converts the
        dictionary and forwards the Config to ``bench_fn``.

    Notes:
        ``n_runs`` is accepted for proposer compatibility but ignored.  The
        adapter performs no database access itself; caching remains entirely in
        ``bench_fn``.  Conversion and benchmark exceptions propagate so the
        proposer can mark that candidate as infinite latency.
    """

    def adapted(config_dict: Dict[str, Any], n_runs=None) -> List[float]:
        """Convert and benchmark one proposer candidate; ``n_runs`` is ignored."""
        config = to_config(config_dict)
        return bench_fn(config)

    return adapted


@LibTuner.register_policy("flagtune")
def flagtune_policy(
    self,
    bench_fn: Callable[[triton.Config], List[float]],
    configs: Iterator[triton.Config],
    args: Tuple[Any],
    kwargs: Dict[str, Any],
) -> Tuple[triton.Config, Dict[str, float]]:
    """Select a config through FlagTree's XGBoost and genetic proposer.

    Args:
        self: Active ``LibTuner`` containing legacy and pair routing metadata
            and normalized runtime arguments in ``self.nargs``.
        bench_fn: Baseline Config benchmark callable, including LibTuner cache
            behavior.
        configs: Baseline configurations.  The iterator is materialized for the
            proposer and otherwise reserved for default-policy fallback.
        args: Positional kernel arguments supplied by the policy interface.
            Shape extraction uses ``self.nargs`` instead.
        kwargs: Runtime keyword arguments.  They are forwarded only when the
            default policy is selected.

    Returns:
        ``(best_config, timings)`` where timings maps successfully benchmarked
        proposed Config objects to their latency values.  Any unavailable,
        disabled, unnamed, ineligible, empty, or failed proposer route returns
        the default policy's result instead.

    Implementation:
        The policy resolves the cached operator/variant proposer, normalizes
        declared shape inputs, adapts ``bench_fn``, then asks the proposer to
        rank XGBoost seeds and generate/benchmark GA candidates.  Returned
        dictionaries become fresh Triton Config objects, inherit the configured
        pre-hook when needed, and are benchmarked to choose the minimum latency.

    Notes:
        FlagGems' legacy ``USE_FLAGTUNE`` path selects expanded configs and
        deliberately uses the default exhaustive LibTuner policy; it is not the
        FlagTree proposer enable switch.  Expected integration failures are
        logged once per reason and do not stop kernel execution.  Candidate
        benchmark failures are skipped individually.  The proposer may invoke
        ``bench_fn`` before the final selection loop, but LibTuner's benchmark
        cache normally prevents duplicate device measurements.
    """
    configs = list(configs)
    if self._flagtune_op_name is not None and runtime.flagtune_enabled(
        self._flagtune_op_name
    ):
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)
    available, exc = _flagtune_available()
    if not available:
        _warn_flagtune_unavailable_once(exc)
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)
    if not _flagtune_enabled():
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)

    op_id = self._flagtune_op_id
    variant = self._flagtune_variant
    if op_id is None or variant is None:
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)
    try:
        from triton.flagtune.identity import (
            ModelIdentity,
            discover_gpu_metadata,
            make_dtype_key,
        )

        arguments = dict(self.nargs or {})
        dtype_resolver = getattr(self, "_flagtune_dtype_resolver", None)
        if dtype_resolver is not None:
            dtypes = tuple(dtype_resolver(arguments))
        else:
            dtypes = _infer_tensor_dtypes(
                arguments[name] for name in self.arg_names if name in arguments
            )
        if not dtypes:
            raise ValueError("no tensor dtypes available for FlagTune identity")
        gpu = discover_gpu_metadata()
        model_identity = ModelIdentity(
            str(gpu["gpu_key"]), op_id, variant, make_dtype_key(dtypes)
        )
    except Exception as exc:
        _warn_flagtune_fallback_once(
            f"identity:{op_id}/{variant}:{exc}",
            "FlagTune identity resolution failed for %s/%s: %s; falling back to default.",
            op_id,
            variant,
            exc,
        )
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)
    identity = model_identity.artifact_key

    try:
        proposer, variant_info = _ensure_flagtune_proposer(model_identity)
    except Exception as exc:
        _warn_flagtune_fallback_once(
            f"init:{identity}",
            "FlagTune proposer init failed for %s: %s; falling back to default.",
            identity,
            exc,
        )
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)

    try:
        shape = variant_info.normalize_inputs(self.nargs)
    except Exception as exc:
        _warn_flagtune_fallback_once(
            f"inputs:{identity}",
            "FlagTune input normalization failed for %s: %s; falling back to default.",
            identity,
            exc,
        )
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)

    param_fields = variant_info.param_names
    to_config = variant_info.to_config
    initial = _configs_to_dicts_for_proposer(configs, param_fields)
    meta = {"op_id": op_id, "variant": variant}

    adapter = _make_proposer_bench_adapter(bench_fn, to_config)

    try:
        result_dicts = proposer(adapter, self.nargs, initial, meta)
    except Exception as exc:
        _warn_flagtune_fallback_once(
            f"run:{identity}:{shape}",
            "FlagTune proposer failed for %s shape=%s: %s; falling back to default.",
            identity,
            shape,
            exc,
        )
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)

    if not result_dicts:
        _warn_flagtune_fallback_once(
            f"empty:{identity}",
            "FlagTune proposer returned empty for %s; falling back to default.",
            identity,
        )
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)

    timings: Dict[triton.Config, float] = {}
    best_config: Optional[triton.Config] = None
    best_latency: float = float("inf")

    for d in result_dicts:
        try:
            cfg = to_config(d)
            if cfg.pre_hook is None and self._flagtune_pre_hook is not None:
                # FlagTune creates fresh Config objects, so it must carry the same
                # TMA pre-hook as expanded FlagGems configs. Without it, the
                # TensorDescriptor block_shape can stay stale while BLOCK_* changes,
                # which makes tl.dot infer a shape different from the accumulator.
                cfg.pre_hook = self._flagtune_pre_hook
            lat = float(bench_fn(cfg)[0])
        except Exception:
            continue
        timings[cfg] = lat
        if lat < best_latency:
            best_latency = lat
            best_config = cfg

    if best_config is None:
        _warn_flagtune_fallback_once(
            f"bench:{identity}:{shape}",
            "FlagTune proposer results all failed benchmark for %s shape=%s; falling back to default.",
            identity,
            shape,
        )
        return LibTuner.get("default").policy(self, bench_fn, configs, args, kwargs)
    return best_config, timings


@LibTuner.register_strategy(None)
@LibTuner.register_strategy("default")
def default_strategy(key: Any) -> Any:
    return key


@LibTuner.register_strategy("log")
def log2_strategy(key: Union[int, float]) -> float:
    return 2 ** math.ceil(math.log2(key))


@LibTuner.register_strategy("align32")
def align32_strategy(key: Union[int, float]) -> int:
    if key == 0:
        return 0
    if key < 32:
        return 2 ** math.ceil(math.log2(key))
    return math.ceil(key / 32) * 32


@LibTuner.register_policy("default")
def default_policy(
    self,
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
    flagtune_op_name=None,
    flagtune_expand_op_name=None,
    flagtune_op_id=None,
    flagtune_variant=None,
    flagtune_yaml_path=None,
    flagtune_pre_hook=None,
    flagtune_dtype_resolver=None,
):
    """Decorator for triton library autotuner.

    `strategy` is a function that takes a key and returns a value.
    It accepts a string, which is the name of a registered strategy, or a callable function.
    In this form it will be applied to each key in the `key` list.
    If it's a tuple or list, it should have the same length as `key`,
    and each element should be a string or a callable function that takes a key and returns a value.
    `policy` accepts a string, which is the name of a registered `LibTuner` subclass, or a `LibTuner` subclass itself.

    FlagTune Args:
        flagtune_op_name: FlagGems legacy runtime enablement key.
        flagtune_expand_op_name: Independent legacy expanded-config name; it
            defaults to ``flagtune_op_name``.
        flagtune_op_id: Globally namespaced FlagTree operator identity.
        flagtune_variant: Single-segment implementation/model variant.
        flagtune_yaml_path: FlagGems expanded-config YAML, not the operator
            registration config.
        flagtune_pre_hook: Hook copied to proposer-created Config objects.
        flagtune_dtype_resolver: Optional trusted code-side callable returning
            tensor dtypes in identity order.

    Returns:
        A decorator that constructs the selected ``LibTuner`` policy class.

    Notes:
        The ``flagtune`` policy uses ``flagtune_op_id`` and ``flagtune_variant``
        for FlagTree model loading while ``flagtune_op_name`` retains legacy FlagGems
        enablement semantics. Missing metadata or integration failures fall
        back to the default policy.  Enabling FlagGems' legacy expanded search
        also intentionally routes through the default policy.
    """

    if isinstance(policy, str):
        policy = LibTuner.get(policy)
    assert issubclass(
        policy, LibTuner
    ), f"the class of {policy.__name__} is {policy.__class__.__name__}, not a subclass of {LibTuner.__name__}"

    def decorator(fn):
        """Construct the selected policy class around a Triton JIT kernel."""
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
            flagtune_op_name=flagtune_op_name,
            flagtune_expand_op_name=flagtune_expand_op_name,
            flagtune_yaml_path=flagtune_yaml_path,
            flagtune_pre_hook=flagtune_pre_hook,
            # Args from new FlagTune
            flagtune_op_id=flagtune_op_id,
            flagtune_variant=flagtune_variant,
            flagtune_dtype_resolver=flagtune_dtype_resolver,
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
        self._has_flagtune_tuner = self._contains_flagtune_tuner(fn)
        self._cpu_cache = dict()

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

    @staticmethod
    def _contains_flagtune_tuner(fn):
        while not isinstance(fn, triton.runtime.JITFunction):
            if (
                getattr(fn, "apply_flagtune", None) is not None
                and getattr(fn, "_flagtune_op_name", None) is not None
            ):
                return True
            fn = getattr(fn, "fn", None)
            if fn is None:
                break
        return False

    def _apply_flagtune(self):
        changed = False
        fn = self.fn
        while not isinstance(fn, triton.runtime.JITFunction):
            apply_flagtune = getattr(fn, "apply_flagtune", None)
            if apply_flagtune is not None:
                changed = apply_flagtune() or changed
            fn = getattr(fn, "fn", None)
            if fn is None:
                break
        if changed:
            for cache in self.kernel_cache:
                cache.clear()

    def key(self, spec_args, dns_args, const_args):
        def spec_arg(arg):
            if hasattr(arg, "data_ptr"):
                if device.vendor_name == "hygon" and hasattr(triton.backends, "hcu"):
                    from triton.backends.hcu.compiler import HIPBackend

                    if hasattr(HIPBackend, "get_tensor_specialization"):
                        return (
                            arg.dtype,
                            arg.data_ptr() % self.divisibility == 0,
                            HIPBackend.get_tensor_specialization(arg),
                        )
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
        if self._has_flagtune_tuner:
            self._apply_flagtune()

        # collect all the arguments
        spec_args = []  # specialize arguments
        dns_args = []  # do not specialize arguments
        const_args = []  # constexpr arguments
        k_args = OrderedDict()
        param_names = list(self.signature.parameters.keys())
        for i, arg in enumerate(args):
            hashable_arg = arg
            if (
                hasattr(arg, "__class__")
                and arg.__class__.__name__ == "TensorDescriptor"
            ):
                # Create a hashable representation of TensorDescriptor
                hashable_arg = (
                    "TensorDescriptor",
                    tuple(arg.shape) if hasattr(arg, "shape") else None,
                    tuple(arg.strides) if hasattr(arg, "strides") else None,
                    tuple(arg.block_shape) if hasattr(arg, "block_shape") else None,
                    arg.padding if hasattr(arg, "padding") else None,
                    # Add other relevant attributes
                )
            if i in self.specialize_indices:
                k_args[param_names[i]] = arg
                spec_args.append(hashable_arg)
            elif i in self.do_not_specialize_indices:
                k_args[param_names[i]] = arg
                dns_args.append(hashable_arg)
            else:
                if major_version == 3 and 3 <= minor_version <= 6:
                    k_args[param_names[i]] = arg
                const_args.append(hashable_arg)
        for p in self.jit_function.params[len(args) :]:
            if p.name in kwargs:
                val = kwargs[p.name]
            elif p.default is inspect._empty:
                continue
            else:
                val = p.default

            if p.is_constexpr:
                const_args.append(val)
                if major_version == 3 and 3 <= minor_version <= 6:
                    k_args[p.name] = val
            elif p.do_not_specialize:
                dns_args.append(val)
                k_args[p.name] = val
            else:
                spec_args.append(val)
                k_args[p.name] = val

        if self._has_flagtune_tuner:
            flagtune_dtypes = _infer_tensor_dtypes(args)
            const_args.append(
                ("flagtune_dtypes",)
                + tuple(str(value) for value in flagtune_dtypes)
            )

        entry_key = self.key(spec_args, dns_args, const_args)
        device = torch_device_fn.current_device()
        # CPU has one device per process and `current_device()` returns the
        # string "cpu" (can't index into the int-keyed `kernel_cache` tuple).
        # This branch is CPU-generic — any future x86 / RISC-V CPU backend
        # reuses the same path; no ARM-specific assumption here.
        if device == "cpu":
            cache = self._cpu_cache
        else:
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
                launch_pre_hooks = []
                while not isinstance(fn, triton.runtime.JITFunction):
                    if isinstance(fn, triton.runtime.Autotuner):
                        config = fn.best_config
                        constexprs["num_warps"] = config.num_warps
                        constexprs["num_stages"] = config.num_stages
                        constexprs["num_ctas"] = config.num_ctas
                        constexprs = {**constexprs, **config.kwargs}
                        tune_constexprs = {**tune_constexprs, **config.kwargs}
                        if config.pre_hook is not None:
                            launch_pre_hooks.append(
                                (config.pre_hook, config.all_kwargs())
                            )
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
                    tuple(launch_pre_hooks),
                )
            return kernel, constexprs

        (
            kernel,
            constexprs,
            tune_constexprs,
            heur_constexprs,
            launch_pre_hooks,
        ) = cache[entry_key]

        if callable(grid):
            # collect all arguments to the grid fn，ie:
            # 1. args,
            # 2. kwargs,
            # 3. all all other captured arguments in CompiledKernel from Autotunner & Heuristics
            # when kwargs & captured args conflict, captured args have higher priority
            meta = {**dict(zip(self.arg_names, args)), **kwargs, **constexprs}
            grid = grid(meta)
        grid = grid + (1, 1)

        if launch_pre_hooks:
            hook_nargs = {**dict(zip(self.arg_names, args)), **kwargs}
            for pre_hook, hook_kwargs in launch_pre_hooks:
                pre_hook({**hook_nargs, **hook_kwargs})

        if major_version == 3 and 3 <= minor_version <= 6:
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


def find_flagtune_benchmark_target(
    public_operator: Callable[..., Any], op_id: str, variant: str
) -> Tuple["LibEntry", LibTuner]:
    """Resolve one benchmarkable LibTuner from public operator metadata.

    Args:
        public_operator: Callable exported from the trusted ``flag_gems`` public
            namespace. Its defining backend module is the only module inspected.
        op_id: Exact globally namespaced operator identity expected on the tuner.
        variant: Exact implementation/model variant expected on the tuner.

    Returns:
        The unique outer :class:`LibEntry` and nested :class:`LibTuner` pair.

    Raises:
        RuntimeError: If no target or multiple distinct tuners match. This makes
            stale YAML/kernel bindings fail before any benchmark state changes.

    Notes:
        YAML never supplies a module or attribute path. The module comes from a
        public callable already selected by FlagGems runtime dispatch.
    """
    module = importlib.import_module(public_operator.__module__)
    candidates: Dict[int, Tuple[LibEntry, LibTuner]] = {}
    for outer in vars(module).values():
        if not isinstance(outer, LibEntry):
            continue
        current = outer.fn
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if (
                isinstance(current, LibTuner)
                and current._flagtune_op_id == op_id
                and current._flagtune_variant == variant
            ):
                candidates[id(current)] = (outer, current)
            current = getattr(current, "fn", None)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one LibTuner for op_id={op_id!r}, "
            f"variant={variant!r}; found {len(candidates)}"
        )
    return next(iter(candidates.values()))


def clear_libentry_dispatch_cache(entry: "LibEntry") -> None:
    """Clear one LibEntry's device and CPU dispatch caches before benchmarking.

    Args:
        entry: Trusted :class:`LibEntry` returned by
            :func:`find_flagtune_benchmark_target`.

    Returns:
        ``None``.  GPU kernel-cache mappings and the CPU dispatch mapping are
        mutated in place.

    Usage:
        The generic FlagTune worker calls this before each shape so the public
        operator re-enters dispatch and invokes the currently configured tuner
        instead of returning a wrapper cached by an earlier shape.

    Notes:
        This helper does not clear compiled Triton binaries, LibTuner's SQLite
        config/benchmark caches, or global FlagTune proposer pools.  It is meant
        only for controlled offline benchmarking, not concurrent production
        dispatch.
    """
    for cache in entry.kernel_cache:
        cache.clear()
    entry._cpu_cache.clear()


def libentry():
    """Decorator for triton library entries."""

    def decorator(fn):
        return LibEntry(fn)

    return decorator
