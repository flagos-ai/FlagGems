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

import copy
import ctypes
import importlib

import pytest
import torch

import flag_gems


@pytest.fixture(scope="module")
def hygon_mm():
    if flag_gems.vendor_name != "hygon":
        pytest.skip("Hygon MM configuration tests")
    pytest.importorskip("triton.backends.hcu.compiler")
    module = importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm")
    if not module._HYGON_MM_PACKED_ASYNC_OPTIONS_SUPPORTED:
        pytest.skip("Packed transposed HCU options are unavailable")
    return module


def test_hygon_mm_option_schema_and_shared_configs(hygon_mm):
    original = hygon_mm.runtime.get_tuned_config("mm")
    before = [copy.deepcopy(config.all_kwargs()) for config in original]
    configs = hygon_mm._hygon_mm_configs()
    assert [config.all_kwargs() for config in original] == before
    assert len({frozenset(config.kwargs) for config in configs}) == 1
    assert all("ASYNC_LOAD" in config.kwargs for config in configs)
    packed = [
        config
        for config in configs
        if hygon_mm._hygon_mm_is_packed_async_config(config)
    ]
    assert {config.kwargs["BLOCK_M"] for config in packed} == {64, 128}
    assert all(config.kwargs["kpack"] == 2 for config in packed)
    large_async = [
        config
        for config in configs
        if config.kwargs["ASYNC_LOAD"] and hygon_mm._hygon_mm_is_large_config(config)
    ]
    assert len(large_async) == 2
    assert {config.kwargs["GROUP_M"] for config in large_async} == {1, 4}
    assert all(config.kwargs["kpack"] == 1 for config in large_async)


@pytest.mark.parametrize(
    "M,N,K,large,masked,allow_async",
    [
        (1024, 1024, 1024, True, False, True),
        (256, 256, 48, True, False, True),
        (256, 256, 16, True, False, True),
        (1024, 1024, 1024, False, False, False),
        (1025, 1024, 1024, True, True, False),
        (1024, 1025, 1024, True, True, False),
        (65, 97, 67, False, True, False),
    ],
)
def test_hygon_mm_async_pruning(hygon_mm, M, N, K, large, masked, allow_async):
    configs = hygon_mm._hygon_mm_edge_prune(
        hygon_mm._hygon_mm_configs(),
        dict(M=M, N=N, K=K, LARGE_TILE_CONFIGS=large),
        MASK_EDGES=masked,
    )
    packed = [config for config in configs if config.kwargs["ASYNC_LOAD"]]
    assert bool(packed) == allow_async
    assert any(not config.kwargs["ASYNC_LOAD"] for config in configs)
    for config in packed:
        assert M % config.kwargs["BLOCK_M"] == 0
        assert N % config.kwargs["BLOCK_N"] == 0


@pytest.mark.parametrize("key_limit", [1, 64])
@pytest.mark.parametrize("family", ["packed", "large"])
def test_hygon_mm_async_cache_roundtrip(hygon_mm, tmp_path, key_limit, family):
    from flag_gems.utils.models.sql import SQLPersistantModel

    model = SQLPersistantModel(f"sqlite:///{tmp_path / 'mm.sqlite'}")
    model.key_count_limit = key_limit
    config = next(
        config
        for config in hygon_mm._hygon_mm_configs()
        if config.kwargs["ASYNC_LOAD"]
        and (
            config.kwargs["BLOCK_M"] == 256
            if family == "large"
            else hygon_mm._hygon_mm_is_packed_async_config(config)
        )
    )
    sync = copy.deepcopy(config)
    sync.kwargs["ASYNC_LOAD"] = False
    keys = [1024, 1024, 1024]
    name = f"hygon-mm-benchmark-{family}-{key_limit}"
    try:
        model.put_benchmark(name, keys, config, (1.0, 0.9, 1.1))
        model.put_benchmark(name, keys, sync, (2.0, 1.9, 2.1))
        assert model.get_benchmark(name, keys, config) == (1.0, 0.9, 1.1)
        assert model.get_benchmark(name, keys, sync) == (2.0, 1.9, 2.1)
        choice = f"hygon-mm-choice-{family}-{key_limit}"
        model.put_config(choice, keys, config)
        restored = model.get_config(choice, keys)
        assert restored is not None
        assert restored.kwargs["ASYNC_LOAD"] == 1
        assert hygon_mm._hygon_mm_is_dense_config(restored)
        if family == "large":
            assert hygon_mm._hygon_mm_is_large_config(restored)
        else:
            assert hygon_mm._hygon_mm_is_packed_async_config(restored)
        assert SQLPersistantModel.get_config_dict(restored) == (
            SQLPersistantModel.get_config_dict(config)
        )
    finally:
        model.engine.dispose()


@pytest.mark.parametrize(
    "M,N,K,large,masked,expected",
    [
        (256, 256, 16, True, False, 2),
        (128, 128, 128, True, False, 0),
        (256, 256, 128, True, True, 0),
        (1024, 1024, 1024, False, False, 0),
    ],
)
def test_hygon_mm_large_async_domain(hygon_mm, M, N, K, large, masked, expected):
    configs = hygon_mm._hygon_mm_edge_prune(
        hygon_mm._hygon_mm_configs(),
        dict(M=M, N=N, K=K, LARGE_TILE_CONFIGS=large),
        MASK_EDGES=masked,
    )
    choices = [
        c for c in configs if c.kwargs["ASYNC_LOAD"] and c.kwargs["BLOCK_M"] == 256
    ]
    assert len(choices) == expected
    assert all(hygon_mm._hygon_mm_is_large_config(c) for c in choices)


def _expanded_configs(module):
    return module.runtime.ops_get_configs(
        "mm_hygon", yaml_path=module.EXPAND_CONFIG_FILENAME
    )


def test_hygon_mm_expanded_regular_geometry_and_schema(hygon_mm):
    from itertools import product

    configs = _expanded_configs(hygon_mm)
    identities = {
        (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages) for c in configs
    }
    assert configs and len(identities) == len(configs)
    assert len({frozenset(c.kwargs) for c in configs}) == 1
    assert all("PROFILE" not in c.kwargs for c in configs)
    assert {c.kwargs["LOAD_MODE"] for c in configs} == {0, 1, 2, 3}
    assert {c.num_warps for c in configs} == {1, 2, 4, 8}
    assert {c.num_stages for c in configs} == {1, 2, 3, 4}
    assert {c.kwargs["kpack"] for c in configs} == {1, 2}
    assert {c.kwargs["sched_latency"] for c in configs} == {"none"}
    # Every geometry retains a plain synchronous fallback. The profile space
    # replaces the old winner-preserving Cartesian grid; it is not its subset.
    fallback = {
        (
            c.kwargs["BLOCK_M"],
            c.kwargs["BLOCK_N"],
            c.kwargs["BLOCK_K"],
            c.kwargs["GROUP_M"],
        )
        for c in configs
        if c.num_stages == 1 and c.kwargs["LOAD_MODE"] == 0
    }
    assert fallback == set(
        product(
            (16, 32, 64, 128, 256), (16, 32, 64, 128, 256), (16, 32, 64, 128), (1, 4, 8)
        )
    )


@pytest.mark.parametrize(
    "load_mask,expected", [(0, {0}), (1, {0, 1}), (3, {0, 1, 2, 3})]
)
def test_hygon_mm_expanded_load_modes_not_shape_ranked(hygon_mm, load_mask, expected):
    configs = _expanded_configs(hygon_mm)
    for M, N, K, masked in [
        (64, 128, 48, False),
        (4096, 4096, 4096, False),
        (257, 129, 144, True),
    ]:
        pruned = hygon_mm._hygon_mm_edge_prune(
            configs,
            dict(M=M, N=N, K=K, ASYNC_LOAD_MASK=load_mask),
            LARGE_TILE_CONFIGS=False,
            MASK_EDGES=masked,
        )
        matched = [
            c
            for c in pruned
            if c.num_warps == 4
            and c.num_stages == 2
            and c.kwargs.get("sync_lds_buffers", 1) == 1
            and all(
                c.kwargs[k] == v
                for k, v in dict(
                    BLOCK_M=64,
                    BLOCK_N=128,
                    BLOCK_K=32,
                    GROUP_M=1,
                    mmac_transposed=True,
                    optimize_epilogue=True,
                    schedule_hint="local-prefetch",
                    kpack=1,
                    waves_per_eu=1,
                ).items()
            )
        ]
        assert {c.kwargs["LOAD_MODE"] for c in matched} == expected


def test_hygon_mm_expanded_prune_tail_and_async_contracts(hygon_mm):
    configs = _expanded_configs(hygon_mm)
    args = dict(M=256, N=256, K=16, ASYNC_LOAD_MASK=3)
    shallow = hygon_mm._hygon_mm_edge_prune(configs, args, MASK_EDGES=False)
    deep = hygon_mm._hygon_mm_edge_prune(configs, dict(args, K=16384), MASK_EDGES=False)
    assert [c.all_kwargs() for c in shallow] == [c.all_kwargs() for c in deep]
    for config in shallow:
        c = config.kwargs
        assert 256 % c["BLOCK_M"] == 256 % c["BLOCK_N"] == 0
        assert not c["LOAD_MODE"] or config.num_stages in (2, 3, 4)
    assert any(c.num_stages == 3 for c in shallow)


def test_hygon_mm_expanded_resolver_and_capability_cache_key(hygon_mm):
    tuner = hygon_mm.mm_kernel
    while not hasattr(tuner, "_flagtune_expand_op_name"):
        tuner = tuner.fn
    assert tuner._flagtune_expand_op_name == "mm_hygon"
    assert "ASYNC_LOAD_MASK" in tuner.keys
    assert "K_DIVISIBLE_BY_16" in tuner.keys
    configs, strategy = tuner._flagtune_configs_for_mode(
        "mm", hygon_mm.runtime.TuningMode.EXPANDED
    )
    assert [c.all_kwargs() for c in configs] == [
        c.all_kwargs() for c in _expanded_configs(hygon_mm)
    ]
    assert len(strategy) == len(tuner.keys) == 8
    assert strategy == ["default"] * 8


@pytest.fixture
def cpu_mv_raw_write(monkeypatch):
    module = importlib.import_module("flag_gems.runtime.backend._hygon.ops.mv")

    def write(inp, vec, out):
        # Emulate a Triton pointer write without PyTorch mutation tracking.
        # This isolates wrapper contracts from GPU compilation and tuning.
        value = torch.mv(inp, vec).contiguous()
        assert out.device.type == "cpu" and out.is_contiguous()
        assert out.device == value.device and out.dtype == value.dtype
        assert out.numel() == value.numel()
        ctypes.memmove(
            out.data_ptr(), value.data_ptr(), out.numel() * out.element_size()
        )
        return out

    monkeypatch.setattr(module, "_mv_out", write)
    monkeypatch.setattr(module, "mv", torch.mv)


@pytest.mark.parametrize("M,N", [(1, 4), (4, 1)])
def test_hygon_mm_mv_out_updates_saved_tensor_version(hygon_mm, cpu_mv_raw_write, M, N):
    a = torch.ones(M, 8, device="cpu", dtype=torch.float32)
    b = torch.ones(8, N, device="cpu", dtype=torch.float32)
    out = torch.zeros(M, N, device="cpu", dtype=torch.float32)
    x = torch.ones_like(out, requires_grad=True)
    loss = (x * out).sum()
    version = out._version
    assert hygon_mm.mm_out(a, b, out=out) is out
    assert torch.equal(out, torch.mm(a, b))
    assert out._version == version + 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("M,N", [(1, 4), (4, 1)])
@pytest.mark.parametrize("view", [False, True])
def test_hygon_mm_mv_out_preserves_grad_rejection(
    hygon_mm, cpu_mv_raw_write, M, N, view
):
    a = torch.ones(M, 8, device="cpu", dtype=torch.float32)
    b = torch.ones(8, N, device="cpu", dtype=torch.float32)
    base = torch.zeros(M * N, device="cpu", dtype=torch.float32, requires_grad=True)
    if view:
        out = base.view(M, N)
    else:
        out = torch.zeros(M, N, device="cpu", dtype=torch.float32, requires_grad=True)
    before = out.detach().clone()
    with pytest.raises(RuntimeError):
        hygon_mm.mm_out(a, b, out=out)
    assert torch.equal(out, before)


@pytest.mark.parametrize("M,N", [(1, 4), (4, 1)])
def test_hygon_mm_mv_out_preserves_inference_rules(hygon_mm, cpu_mv_raw_write, M, N):
    a = torch.ones(M, 8, device="cpu", dtype=torch.float32)
    b = torch.ones(8, N, device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        out = torch.zeros(M, N, device="cpu", dtype=torch.float32)
        assert hygon_mm.mm_out(a, b, out=out) is out
        assert torch.equal(out, torch.mm(a, b))
    with pytest.raises(RuntimeError):
        hygon_mm.mm_out(a, b, out=out)


@pytest.fixture(scope="module")
def hygon_splitk(hygon_mm):
    return importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm_splitk")


def test_hygon_small_mm_expanded_regular_partition_profiles(hygon_splitk):
    from itertools import product

    configs = hygon_splitk.runtime.ops_get_configs(
        "mm_hygon_splitk", yaml_path=hygon_splitk.EXPAND_CONFIG_FILENAME
    )

    def key(c):
        meta = dict(c.kwargs)
        meta.setdefault("wave_private_sync", False)
        return tuple(sorted(meta.items())), c.num_warps, c.num_stages

    assert len(configs) == len({key(c) for c in configs})
    assert len({frozenset(c.kwargs) for c in configs}) == 1
    assert all("PROFILE" not in c.kwargs for c in configs)
    assert {key(c) for c in hygon_splitk._splitk_default_configs()} <= {
        key(c) for c in configs
    }
    geometry = set(product((16, 32, 64), (16, 32, 64), (16, 32, 64)))
    global_configs = [c for c in configs if c.kwargs["KIND"] == 0]
    local_configs = [c for c in configs if c.kwargs["KIND"] == 1]
    assert {
        (c.kwargs["BM"], c.kwargs["BN"], c.kwargs["BK"], c.kwargs["SPLITS"])
        for c in global_configs
    } == {(*tile, splits) for tile in geometry for splits in (1, 2, 4, 8, 16, 32, 64)}
    assert {
        (
            c.kwargs["BM"],
            c.kwargs["BN"],
            c.kwargs["BK"],
            c.kwargs["SPLITS"],
            c.num_stages,
        )
        for c in local_configs
    } == {
        (*tile, splits, stages)
        for tile in geometry
        for splits in (2, 4, 8)
        for stages in (1, 2)
    }
    assert all(
        c.num_warps == c.kwargs["SPLITS"]
        and c.kwargs["wave_private_sync"]
        and c.kwargs["optimize_epilogue"]
        and c.kwargs["LOAD_MODE"] == 0
        for c in local_configs
    )
    assert all(
        not c.kwargs["wave_private_sync"] and not c.kwargs["optimize_epilogue"]
        for c in global_configs
    )
    # FP32 global partial stores and FP16/BF16 local reductions have different
    # epilogue policies. Keep both synchronous global producer tiers and async B.
    for bm, bn, bk in geometry:
        for splits in (1, 2, 4, 8, 16, 32, 64):
            members = [
                c
                for c in global_configs
                if tuple(c.kwargs[k] for k in ("BM", "BN", "BK", "SPLITS"))
                == (bm, bn, bk, splits)
            ]
            assert {
                (c.kwargs["LOAD_MODE"], c.num_stages, c.num_warps) for c in members
            } == {(0, 1, 1 if bm * bn == 256 else 2), (0, 2, 2), (0, 2, 4), (2, 3, 2)}
    assert all(c.kwargs["schedule_hint"] == "local-prefetch" for c in configs)


@pytest.mark.parametrize(
    "mask,expected_modes", [(0, {0}), (1, {0}), (2, {0, 2}), (3, {0, 2})]
)
def test_hygon_small_mm_prunes_capability_only(hygon_splitk, mask, expected_modes):
    configs = hygon_splitk.runtime.ops_get_configs(
        "mm_hygon_splitk", yaml_path=hygon_splitk.EXPAND_CONFIG_FILENAME
    )
    tail = hygon_splitk._splitk_prune(
        configs,
        dict(M=33, N=48, K=4112),
        ASYNC_LOAD_MASK=mask,
        WAVE_PRIVATE_SYNC_SUPPORTED=True,
    )
    deep = hygon_splitk._splitk_prune(
        configs,
        dict(
            M=128, N=64, K=16384, ASYNC_LOAD_MASK=mask, WAVE_PRIVATE_SYNC_SUPPORTED=True
        ),
    )
    assert [c.all_kwargs() for c in tail] == [c.all_kwargs() for c in deep]
    assert {c.kwargs["KIND"] for c in tail} == {0, 1}
    assert {c.kwargs["LOAD_MODE"] for c in tail} == expected_modes
    assert [c.all_kwargs() for c in tail] == [
        c.all_kwargs() for c in configs if c.kwargs["LOAD_MODE"] in expected_modes
    ]


def test_hygon_small_mm_resolver_and_timed_post(hygon_splitk):
    tuner = hygon_splitk._mm_splitk_partial.fn
    assert tuner._flagtune_op_name == "mm"
    assert tuner._flagtune_expand_op_name == "mm_hygon_splitk"
    assert tuner.keys == [
        "M",
        "N",
        "K",
        "stride_bk",
        "stride_bn",
        "B_K_CONTIGUOUS",
        "ASYNC_LOAD_MASK",
        "WAVE_PRIVATE_SYNC_SUPPORTED",
        "M_ALIGNMENT",
        "N_ALIGNMENT",
    ]
    configs, strategy = tuner._flagtune_configs_for_mode(
        "mm", hygon_splitk.runtime.TuningMode.EXPANDED
    )
    assert [c.all_kwargs() for c in configs] == [
        c.all_kwargs()
        for c in hygon_splitk.runtime.ops_get_configs(
            "mm_hygon_splitk", yaml_path=hygon_splitk.EXPAND_CONFIG_FILENAME
        )
    ]
    assert strategy == ["default"] * len(tuner.keys)
    # Exercise normalization after the actual Expanded mode switch.
    assert len(tuner._normalize_strategy(strategy)) == len(tuner.keys)
    assert tuner.post_hook is hygon_splitk._splitk_post_hook


@pytest.mark.parametrize("transpose_b", [False, True])
def test_hygon_small_mm_uses_launch_metadata_for_cached_post(
    hygon_splitk, monkeypatch, transpose_b
):
    import contextlib

    m = hygon_splitk
    hy = importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm")
    calls = []
    posts = []
    monkeypatch.setattr(m, "_splitk_parameters", lambda a, b, c: (32, 32, 32, 16))
    monkeypatch.setattr(hy, "_hygon_mm_async_load_mask", lambda *a: 1)
    monkeypatch.setattr(m.torch_device_fn, "device", lambda d: contextlib.nullcontext())
    metas = [
        dict(BM=16, BN=16, SPLITS=4, KIND=1, POST_BLOCK=256),
        dict(BM=32, BN=32, SPLITS=32, KIND=0, POST_BLOCK=128),
    ]

    class Entry:
        best_config = object()

        def __getitem__(self, grid):

            def launch(*args, **kw):
                meta = metas[len(calls)]
                calls.append((grid(meta), args, kw))
                return (object(), meta)

            return launch

    monkeypatch.setattr(m, "_mm_splitk_partial", Entry())
    monkeypatch.setattr(
        m, "_splitk_post_hook", lambda args, exc: posts.append((args, exc))
    )
    a = torch.empty((33, 4112), dtype=torch.bfloat16)
    b = (
        torch.empty((48, 4112), dtype=torch.bfloat16).t()
        if transpose_b
        else torch.empty((4112, 48), dtype=torch.bfloat16)
    )
    c = torch.empty((33, 48), dtype=torch.bfloat16)
    assert m.try_mm_splitk(a, b, c) and m.try_mm_splitk(a, b, c)
    assert calls[0][0] == (3, 3, 1) and calls[1][0] == (2, 2, 32)
    for i, (grid, args, kw) in enumerate(calls):
        assert (
            args[0] is a
            and args[1] is b
            and (args[3] is c)
            and (tuple(args[2].shape) == (64, 33, 48))
        )
        assert args[7:] == b.stride()
        assert kw == dict(
            B_K_CONTIGUOUS=transpose_b,
            ASYNC_LOAD_MASK=1,
            WAVE_PRIVATE_SYNC_SUPPORTED=m._SPLITK_WAVE_PRIVATE_SYNC_SUPPORTED,
            M_ALIGNMENT=1,
            N_ALIGNMENT=16,
        )
        assert (
            posts[i][0]["SPLITS"] == metas[i]["SPLITS"]
            and posts[i][0]["KIND"] == metas[i]["KIND"]
            and (posts[i][1] is None)
        )


@pytest.mark.parametrize("mask", [0, 3])
def test_hygon_small_mm_older_backend_keeps_global_profiles(hygon_splitk, mask):
    configs = hygon_splitk.runtime.ops_get_configs(
        "mm_hygon_splitk", yaml_path=hygon_splitk.EXPAND_CONFIG_FILENAME
    )
    before = [copy.deepcopy(c.all_kwargs()) for c in configs]
    pruned = hygon_splitk._splitk_prune(
        configs, {}, ASYNC_LOAD_MASK=mask, WAVE_PRIVATE_SYNC_SUPPORTED=False
    )
    expected = []
    for c in configs:
        if c.kwargs["KIND"] == 0 and not c.kwargs["LOAD_MODE"] & ~mask:
            meta = dict(c.all_kwargs())
            meta.pop("wave_private_sync")
            expected.append(meta)
    assert [c.all_kwargs() for c in pruned] == expected
    assert all(
        c.kwargs["KIND"] == 0 and "wave_private_sync" not in c.kwargs for c in pruned
    )
    assert before == [c.all_kwargs() for c in configs]


def test_hygon_small_mm_post_is_independently_expanded(hygon_splitk):
    tuner = hygon_splitk._mm_splitk_post.fn
    configs, strategy = tuner._flagtune_configs_for_mode(
        "mm", hygon_splitk.runtime.TuningMode.EXPANDED
    )
    assert tuner._flagtune_expand_op_name == "mm_hygon_splitk_post"
    assert tuner.keys == ["MN", "SPLITS"] and strategy == ["default"] * 2
    assert len(configs) == 2 and {c.kwargs["BLOCK"] for c in configs} == {128, 256}
    defaults, _ = tuner._flagtune_configs_for_mode(
        "mm", hygon_splitk.runtime.TuningMode.DEFAULT
    )
    assert len(defaults) == 1 and defaults[0].kwargs["BLOCK"] == 256


def test_hygon_small_mm_timed_post_grid_uses_post_metadata(hygon_splitk, monkeypatch):
    calls = []

    class Entry:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid({"BLOCK": 128}), grid({"BLOCK": 256}), args, kwargs))

            return launch

    monkeypatch.setattr(hygon_splitk, "_mm_splitk_post", Entry())
    args = dict(W=object(), C=object(), M=33, N=48, KIND=0, SPLITS=16)
    hygon_splitk._splitk_post_hook(args, None)
    assert calls == [((13,), (7,), (args["W"], args["C"], 1584, 16), {})]
    hygon_splitk._splitk_post_hook(dict(args, KIND=1), None)
    hygon_splitk._splitk_post_hook(dict(args, SPLITS=1), None)
    hygon_splitk._splitk_post_hook(args, RuntimeError("skip failed primary"))
    assert len(calls) == 1


def test_hygon_mm_named_profiles_have_distinct_representatives(hygon_mm):
    from triton.backends.hcu.compiler import HIPOptions

    from flag_gems.runtime.backend._hygon.gemm_config_profiles import build_mm_configs

    names = [
        "generic_min",
        "generic_pipe",
        "sync",
        "packed",
        "register_prefetch",
        "shared_prefetch",
        "async_a",
        "async_b",
        "async_pair",
        "async_deep",
    ]
    if "sync_lds_buffers" not in HIPOptions.__dataclass_fields__:
        names.remove("shared_prefetch")

    def hook(values):
        return None

    representatives = []
    for name in names:
        configs = build_mm_configs(
            dict(
                PROFILE=[name],
                BLOCK_M=[64],
                BLOCK_N=[64],
                BLOCK_K=[32],
                GROUP_M=[1],
                s=[1, 2, 3, 4],
                w=[1, 2, 4, 8],
            ),
            pre_hook=hook,
        )
        assert len(configs) == 1 and configs[0].pre_hook is hook
        representatives.append(tuple(sorted(configs[0].all_kwargs().items())))
    assert len(set(representatives)) == len(names)


def test_hygon_mm_expanded_older_compiler_omits_ring_option(hygon_mm, monkeypatch):
    from triton.backends.hcu.compiler import HIPOptions

    if "sync_lds_buffers" not in HIPOptions.__dataclass_fields__:
        pytest.skip("Compiler already lacks the optional ring setting")

    modern = _expanded_configs(hygon_mm)
    expected = []
    for config in modern:
        if config.kwargs["sync_lds_buffers"] == 1:
            spec = config.all_kwargs()
            spec.pop("sync_lds_buffers")
            expected.append(spec)
    fields = dict(HIPOptions.__dataclass_fields__)
    fields.pop("sync_lds_buffers")
    with monkeypatch.context() as patch:
        patch.setattr(HIPOptions, "__dataclass_fields__", fields)
        legacy = _expanded_configs(hygon_mm)
    assert len(legacy) == len(expected)
    assert [config.all_kwargs() for config in legacy] == expected
    assert all("sync_lds_buffers" not in config.kwargs for config in legacy)


def test_hygon_mm_scalar_copy_alias_needs_retained_control(hygon_mm):
    from triton.backends.hcu.compiler import HIPOptions

    from flag_gems.runtime.backend._hygon.gemm_config_profiles import build_mm_configs

    ranges = dict(BLOCK_M=[16], BLOCK_N=[64], BLOCK_K=[16], GROUP_M=[1], s=[2], w=[4])
    cases = [
        (["async_a"], 1),
        (["sync", "async_a"], 0),
        (["async_pair"], 3),
        (["async_b", "async_pair"], 2),
    ]
    for names, mode in cases:
        configs = build_mm_configs(dict(ranges, PROFILE=names))
        assert len(configs) == 1 and configs[0].kwargs["LOAD_MODE"] == mode
    if "sync_lds_buffers" in HIPOptions.__dataclass_fields__:
        assert len(build_mm_configs(dict(ranges, PROFILE=["shared_prefetch"]))) == 1
        assert (
            len(build_mm_configs(dict(ranges, PROFILE=["sync", "shared_prefetch"])))
            == 1
        )


def test_hygon_mm_avoids_uniform_buffer_footprint_pruning(hygon_mm):
    configs = _expanded_configs(hygon_mm)
    # Large synchronous tiles can use single-buffer or non-pipelined fallback
    # on irregular inputs; the async allocation bound must not prune them.
    large = [
        c
        for c in configs
        if c.kwargs["BLOCK_M"] == c.kwargs["BLOCK_N"] == 256
        and c.kwargs["BLOCK_K"] == 128
        and c.kwargs["LOAD_MODE"] == 0
        and c.kwargs.get("sync_lds_buffers", 1) == 1
    ]
    assert {c.num_stages for c in large} == {1, 2, 3}
    # With weak N, async A can retain two A planes and one B conversion plane.
    # A uniform two-plane estimate for both operands would wrongly remove it.
    pruned = hygon_mm._hygon_mm_edge_prune(
        configs, dict(M=35, N=81, K=2048, ASYNC_LOAD_MASK=1), MASK_EDGES=True
    )
    assert any(
        c.kwargs["BLOCK_M"] == 16
        and c.kwargs["BLOCK_N"] == 128
        and c.kwargs["BLOCK_K"] == 128
        and c.kwargs["LOAD_MODE"] == 1
        and c.num_stages == 2
        for c in pruned
    )


def test_hygon_mv_expanded_keeps_register_and_occupancy_boundaries(hygon_mm):
    configs = hygon_mm.runtime.ops_get_configs("mv_hygon")
    members = {(c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.num_warps) for c in configs}
    # Preserve the largest admitted accumulation state, but use more waves
    # when doubling its geometry. Keep the independently useful extra-wave
    # tier on small tiles even when it owns no additional elements.
    assert (1024, 8, 1) in members
    assert (2048, 8, 1) not in members and (2048, 8, 2) in members
    assert (256, 1, 4) in members and (256, 1, 8) in members
    assert (32, 1, 1) in members and (32, 1, 2) in members
    assert (32, 1, 4) not in members
    assert all(c.num_stages == 2 for c in configs)
    for bm, bn, waves in members:
        assert {
            c.kwargs["LOOP_STAGES"]
            for c in configs
            if (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.num_warps)
            == (bm, bn, waves)
        } == {1, 2, 3}


def test_hygon_mv_expanded_preserves_pre_hook(hygon_mm):
    def hook(values):
        return None

    original = hygon_mm.runtime.ops_get_configs("mv_hygon")
    hooked = hygon_mm.runtime.ops_get_configs("mv_hygon", pre_hook=hook)
    assert [c.all_kwargs() for c in hooked] == [c.all_kwargs() for c in original]
    assert all(c.pre_hook is hook for c in hooked)


def test_hygon_combined_expanded_budget(hygon_mm, hygon_splitk):
    packet = importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm_packet")
    partial = hygon_splitk.runtime.ops_get_configs(
        "mm_hygon_splitk", yaml_path=hygon_splitk.EXPAND_CONFIG_FILENAME
    )
    post = hygon_splitk.runtime.ops_get_configs(
        "mm_hygon_splitk_post", yaml_path=hygon_splitk.EXPAND_CONFIG_FILENAME
    )
    # Count the two storage layouts separately, plus MV's single MMAC family.
    # The ordinary MM/MV candidates are already counted in their YAML pools.
    python_candidates = (
        sum(len(packet._packet_variants(nt=nt)) for nt in (False, True)) + 1
    )
    total = (
        len(_expanded_configs(hygon_mm))
        + len(partial)
        + len(post)
        + len(hygon_mm.runtime.ops_get_configs("mv_hygon"))
        + python_candidates
    )
    assert total <= 4000
