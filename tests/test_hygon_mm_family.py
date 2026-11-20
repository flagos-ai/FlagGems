"""Real family router cache, graph-capture and dynamic operand contracts."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="requires the Hygon backend"
)


@pytest.fixture
def router(monkeypatch):
    module = importlib.import_module("flag_gems.runtime.backend._hygon.ops.mm")
    module.clear_mm_family_choices()
    monkeypatch.setattr(module.torch_device_fn, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield module
    module.clear_mm_family_choices()


def tensor(shape, stride, pointer, dtype=torch.bfloat16):
    return SimpleNamespace(
        ndim=2,
        shape=shape,
        stride=lambda dim=None: stride if dim is None else stride[dim],
        dtype=dtype,
        device=torch.device("cuda:0"),
        data_ptr=lambda: pointer,
        untyped_storage=lambda: SimpleNamespace(
            data_ptr=lambda: pointer, nbytes=lambda: 1 << 20
        ),
        requires_grad=False,
        is_inference=lambda: False,
    )


def operands(offset=0, dtype=torch.bfloat16):
    return (
        tensor((35, 512), (512, 1), 0x100000 + offset, dtype),
        tensor((512, 80), (80, 1), 0x100000000 + offset, dtype),
        tensor((35, 80), (80, 1), 0x200000000 + offset, dtype),
    )


def install_benchmark(router, monkeypatch):
    measured = []

    def bench(fn, **kwargs):
        measured.append(kwargs)
        return float(fn())

    monkeypatch.setattr(router, "gemm_replay_benchmarker", lambda *_: bench)
    return measured


def test_family_hot_cache_uses_current_tensors_and_skips_pool_rebuild(
    router, monkeypatch
):
    measured = install_benchmark(router, monkeypatch)
    cfg = router.SharedMMConfig((64, 128, 32))
    enumerated, launched = [], []

    def candidates(*args, **kwargs):
        enumerated.append(args)
        return (cfg,)

    monkeypatch.setattr(router, "mm_family_candidates", candidates)

    def launch(a, b, c, config):
        launched.append((a, b, c, config))
        return 1

    monkeypatch.setattr(router, "_dispatch_hygon_mm_candidate", launch)
    a, b, c = operands()
    assert router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    first_measurements = len(measured)
    new = operands(offset=16)
    assert router.try_mm_family(*new, lambda: 10, tuning_key=("default",))
    assert len(enumerated) == 1 and len(measured) == first_measurements
    assert all(x is y for x, y in zip(launched[-1][:3], new))
    # A warm graph uses exactly the cached decision without running a tuner.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert router.try_mm_family(*new, lambda: 10, tuning_key=("default",))
    assert len(enumerated) == 1 and len(measured) == first_measurements


def test_family_cold_graph_never_tunes(router, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cold capture must not tune or enumerate kernels")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(router, "mm_family_candidates", forbidden)
    assert not router.try_mm_family(*operands(), forbidden, tuning_key=("default",))
    assert not router._mm_family_choices


@pytest.mark.parametrize(
    "fail_on_cache_hit, clear_on_failure",
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize("candidate_kind", ["shared", "tuned_split_k"])
def test_optional_mm_compile_failure_preserves_incumbent(
    router, monkeypatch, fail_on_cache_hit, clear_on_failure, candidate_kind
):
    from triton.runtime.errors import OutOfResources

    install_benchmark(router, monkeypatch)
    config = (
        router.SharedMMConfig((64, 128, 32))
        if candidate_kind == "shared"
        else router._TUNED_SPLITK_CANDIDATE
    )
    monkeypatch.setattr(router, "mm_family_candidates", lambda *a, **kw: (config,))
    fails = not fail_on_cache_hit
    selected = []

    def optional(*args):
        if fails:
            if clear_on_failure:
                router.clear_mm_family_choices()
            raise OutOfResources(65538, 65536, "shared memory")
        selected.append("shared")
        return 1

    def incumbent():
        selected.append("ordinary")
        return 10

    monkeypatch.setattr(router, "_dispatch_hygon_mm_candidate", optional)
    args = operands()
    assert router.try_mm_family(*args, incumbent, tuning_key=("expanded",))
    assert selected[-1] == ("shared" if fail_on_cache_hit else "ordinary")
    fails = True
    assert router.try_mm_family(*args, incumbent, tuning_key=("expanded",))
    assert selected[-1] == "ordinary"
    choices = router._mm_family_choices._choices
    if clear_on_failure:
        assert not choices
    else:
        assert next(iter(choices.values())).index == 0


def test_offline_mm_tuner_precedes_splitk(router, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline ordinary MM reached Split-K")

    launched = []
    monkeypatch.setattr(router.mm_kernel.fn, "_run_mode", "exhaustive_collection")
    monkeypatch.setattr(router, "try_mm_splitk", forbidden)
    monkeypatch.setattr(router, "_launch_ordinary_mm", lambda *a: launched.append(a))
    args = operands()
    router._dispatch_hygon_mm(*args)
    assert launched == [args]


def test_short_k_sync_admits_odd_n_in_both_layouts(router, monkeypatch):
    m, n, k = 8192, 257, 128
    monkeypatch.setattr(router, "_short_k_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=128),
    )
    a = tensor((m, k), (k, 1), 0x100000)
    c = tensor((m, n), (n, 1), 0x200000000)
    nn = tensor((k, n), (n, 1), 0x100000000)
    nt = tensor((k, n), (1, k), 0x100000000)
    assert router.short_k_mm_candidates(a, nn, c) == router.SHORT_K_MM_CONFIGS
    assert router.short_k_mm_candidates(a, nt, c) == router.SHORT_K_MM_CONFIGS
    assert not router._SHORT_K_STORAGE.supports(a, nn, c)


@pytest.mark.parametrize("k,expected", [(320, (64,)), (512, (64,)), (576, ())])
def test_expanded_short_k_admits_up_to_eight_panels(router, monkeypatch, k, expected):
    m, n = 2048, 2048
    monkeypatch.setattr(router, "_short_k_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=80),
    )
    a = tensor((m, k), (k, 1), 0x100000)
    c = tensor((m, n), (n, 1), 0x200000000)
    for stride in ((n, 1), (1, k)):
        b = tensor((k, n), stride, 0x100000000)
        assert router.short_k_mm_candidates(a, b, c) == ()
        configs = router.short_k_mm_candidates(a, b, c, extended=True)
        assert tuple(config.block_k for config in configs) == expected


@pytest.mark.parametrize(
    "m,n,out_stride,expected",
    [
        (4138, 2048, None, True),
        (32640, 32768, None, True),
        (32767, 32768, None, False),  # Padded tile reaches 2 GiB.
        (32896, 32768, None, False),
        (4138, 2048, (2064, 1), False),
    ],
)
def test_short_k_launch_proves_padded_output_index_range(
    router, monkeypatch, m, n, out_stride, expected
):
    class Entry:
        def __getitem__(self, grid):
            assert grid == (((m + 127) // 128) * ((n + 127) // 128),)
            return lambda *args, **kwargs: kwargs

    monkeypatch.setattr(router, "_ordinary_mm_entry", Entry)
    a = tensor((m, 512), (512, 1), 0x100000)
    c = tensor((m, n), out_stride or (n, 1), 0x200000000)
    for stride in ((n, 1), (1, 512)):
        b = tensor((512, n), stride, 0x100000000)
        launch = router.launch_short_k_mm(a, b, c, router.ShortKMMConfig(64))
        assert launch["USE_I32_INDEX"] is expected


@pytest.mark.parametrize(
    "nt,n,a_offset,b_offset,expected",
    [
        (False, 64, 0, 0, 3),
        (True, 64, 0, 0, 3),
        (False, 65, 0, 0, 1),
        (True, 65, 0, 0, 3),
        (True, 65, 0, 2, 1),
        (True, 65, 2, 0, 2),
    ],
)
def test_ordinary_async_layout_is_per_operand(
    router, monkeypatch, nt, n, a_offset, b_offset, expected
):
    from triton import knobs

    monkeypatch.setattr(router, "_HYGON_MM_LARGE_OPTIONS_SUPPORTED", True)
    monkeypatch.setattr(router, "HAS_HYGON_TLE", True)
    monkeypatch.setattr(
        knobs,
        "hcu",
        SimpleNamespace(
            use_buffer_ops=True,
            use_async_copy=False,
            use_block_pingpong=False,
            use_in_thread_transpose=False,
        ),
    )
    driver = SimpleNamespace(
        get_current_target=lambda: SimpleNamespace(backend="hip", arch="gfx936")
    )
    monkeypatch.setattr(router.triton.runtime.driver, "_active", driver)
    m, k = 64, 128
    a = tensor((m, k), (k, 1), 0x100000 + a_offset)
    b = tensor((k, n), (1, k) if nt else (n, 1), 0x100000000 + b_offset)
    c = tensor((m, n), (n, 1), 0x200000000)
    c.is_contiguous = lambda: True
    for t in (a, b, c):
        t.untyped_storage = lambda: SimpleNamespace(nbytes=lambda: 1 << 20)
    assert router._hygon_mm_async_load_mask(a, b, c, m, n, k) == expected


@pytest.mark.parametrize(
    "m,n,k,nt,expected",
    [
        (2, 64, 4096, True, 4),
        (4, 64, 4096, True, 4),
        (2, 4, 16384, True, 4),
        (2, 64, 4096, False, 0),
        (1, 64, 4096, True, 0),
        (8, 64, 4096, True, 3),
        (4, 2048, 4096, True, 0),
        (63, 4, 16384, True, 4),
        (65, 2, 8192, True, 3),
        (65, 4, 8192, False, 0),
        (161, 4, 16384, True, 1),
        (321, 4, 16384, True, 0),
        (9, 16, 4096, True, 4),
        (16, 16, 4096, True, 4),
        (80, 16, 4096, True, 1),
        (81, 16, 4096, True, 0),
        (5, 33, 4096, True, 4),
        (33, 33, 4096, True, 1),
        (8, 16, 512, True, 0),
        (63, 16, 16384, True, 1),
    ],
)
def test_grouped_mv_candidates_follow_layout_and_parallelism(
    router, m, n, k, nt, expected
):
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if nt else (n, 1), 0x100000000)
    assert len(router.grouped_mv_candidates(a, b, compute_units=80)) == expected


@pytest.mark.parametrize(
    "n,grid_n,full_n", [(3, 2, False), (4, 2, True), (5, 3, False)]
)
@pytest.mark.parametrize("nt", [False, True])
def test_grouped_mv_launch_preserves_tail_mask(
    router, monkeypatch, n, grid_n, full_n, nt
):
    m, k = 4, 128
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if nt else (n, 1), 0x200000)
    c = tensor((m, n), (n, 1), 0x300000)

    class Entry:
        def __getitem__(self, grid):
            assert grid == (grid_n, m)
            return lambda *args, **metadata: metadata

    monkeypatch.setattr(router, "_fixed_mv_entry", Entry)
    opts = router.launch_grouped_mv(a, b, c, router.GroupedMVConfig(2, 128))
    assert opts["FULL_N_TILE"] is full_n


@pytest.mark.parametrize(
    "m,n,expected_tiles",
    [
        (2, 64, ((16, 16), (16, 16))),
        (128, 128, ((16, 16), (16, 16), (32, 32))),
        (4, 2048, ((16, 16), (16, 16), (32, 32), (32, 64, 64))),
    ],
)
def test_splitk_shortlist_uses_tile_parallelism(
    router, monkeypatch, m, n, expected_tiles
):
    driver = SimpleNamespace(
        get_current_target=lambda: SimpleNamespace(backend="hip", arch="gfx936")
    )
    monkeypatch.setattr(router.triton.runtime.driver, "_active", driver)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=128),
    )
    monkeypatch.setattr(router, "_shared_supported", lambda: True)
    monkeypatch.setattr(
        router,
        "local_split_mm_configs",
        lambda: router._SPLITK_LOCAL + (router._SPLITK_LOCAL_WIDE,),
    )
    a = tensor((m, 4096), (4096, 1), 0x100000)
    b = tensor((4096, n), (n, 1), 0x100000000)
    c = tensor((m, n), (n, 1), 0x200000000)
    candidates = router.mm_family_candidates(a, b, c, splitk_pool=True)
    assert tuple(config.tile for config in candidates) == expected_tiles


def test_family_separates_dtype_protocol_and_runtime_eligibility(router, monkeypatch):
    install_benchmark(router, monkeypatch)
    cfg = router.SharedMMConfig((64, 128, 32))
    enumerated = []

    def candidates(a, b, c, **kwargs):
        enumerated.append((a.dtype, kwargs))
        return (cfg,) if cfg.supports(a, b, c) else ()

    monkeypatch.setattr(router, "mm_family_candidates", candidates)
    monkeypatch.setattr(router, "_dispatch_hygon_mm_candidate", lambda *args: 1)
    assert router.try_mm_family(*operands(), lambda: 10, tuning_key=("default",))
    assert router.try_mm_family(
        *operands(dtype=torch.float16), lambda: 10, tuning_key=("default",)
    )
    assert router.try_mm_family(*operands(), lambda: 10, tuning_key=("expanded",))
    assert len(router._mm_family_choices) == 3
    a, b, c = operands()
    c.data_ptr = lambda: 0x200000008
    assert not router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    assert len(enumerated) == 4 and len(router._mm_family_choices) == 3


def test_family_does_not_reuse_legacy_choice_when_storage_alias_changes(
    router, monkeypatch
):
    install_benchmark(router, monkeypatch)

    def candidates(a, b, c, **kwargs):
        return (
            ("legacy",)
            if c.untyped_storage().data_ptr() != a.untyped_storage().data_ptr()
            else ()
        )

    monkeypatch.setattr(router, "mm_family_candidates", candidates)
    monkeypatch.setattr(router, "_dispatch_hygon_mm_candidate", lambda *args: 1)
    a, b, c = operands()
    assert router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    c.untyped_storage = a.untyped_storage
    assert not router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))


def test_family_does_not_reuse_choice_for_different_output_shape(router, monkeypatch):
    install_benchmark(router, monkeypatch)
    cfg = router.SharedMMConfig((64, 128, 32))
    monkeypatch.setattr(
        router,
        "mm_family_candidates",
        lambda a, b, c, **kwargs: (cfg,) if cfg.supports(a, b, c) else (),
    )
    monkeypatch.setattr(router, "_dispatch_hygon_mm_candidate", lambda *args: 1)
    a, b, c = operands()
    assert router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))
    c.shape = (34, 80)
    assert not router.try_mm_family(a, b, c, lambda: 10, tuning_key=("default",))


@pytest.mark.parametrize(
    "m,n,k,layout,expanded,b_offset,expected",
    [
        (4096, 2048, 192, "nn", True, 0, (0, 4)),
        (3072, 2048, 192, "nn", True, 0, (0, 4)),
        (2560, 2048, 192, "nn", True, 0, (0, 4)),
        (2304, 2048, 192, "nn", True, 0, ()),
        (16384, 376, 192, "nn", True, 0, ()),
        (16384, 392, 192, "nn", True, 0, (0, 4)),
        (4097, 2049, 192, "nn", True, 0, ()),
        (4097, 2056, 192, "nn", True, 2, ()),
        (4097, 2056, 193, "nn", True, 0, ()),
        (4097, 2056, 192, "nn", False, 0, ()),
        (512, 257, 192, "nt", False, 0, (0, 4)),
        (65536, 32768, 512, "nn", True, 0, (0, 4)),
        (512, 262152, 4096, "nn", True, 0, (0, 4)),
        (65536, 32768, 512, "nt", False, 0, (0, 4)),
    ],
)
def test_cooperative_shortlist_uses_layout_and_tile_coverage(
    router, monkeypatch, m, n, k, layout, expanded, b_offset, expected
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=128),
    )
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 0x100000000 + b_offset)
    c = tensor((m, n), (n, 1), 0x200000000)
    configs = router._cooperative_mm_candidates(a, b, c, allow_nn=expanded)
    assert tuple(config.group_m for config in configs) == expected


@pytest.mark.parametrize(
    "m,n,compute_units,expected",
    [
        (129, 5120, 80, (4,)),
        (129, 5128, 80, (2, 4)),
        (129, 6912, 80, (2,)),
        (257, 7168, 80, ()),
        (129, 6912, 128, (4,)),
    ],
)
def test_shared_two_cta_shortlist_uses_tile_grid(router, m, n, compute_units, expected):
    configs = router._shared_two_cta_candidates(m, n, compute_units)
    assert tuple(config.num_stages for config in configs) == expected


@pytest.mark.parametrize(
    "m,n,compute_units,eligible",
    [
        (256, 10240, 80, False),
        (257, 10240, 80, True),
        (512, 10240, 80, True),
        (513, 10240, 80, False),
        (257, 10240, 128, False),
    ],
)
def test_shared_one_slot_shortlist_uses_lds_capacity(
    router, m, n, compute_units, eligible
):
    configs = router._shared_one_slot_candidates(m, n, compute_units)
    assert len(configs) == int(eligible)
    for config in configs:
        for transposed in (False, True):
            options = dict(router._shared_compile_parameters(config, transposed))
            assert options["schedule_hint"] == "none"
            assert options["B_K_CONTIGUOUS"] == transposed


@pytest.mark.parametrize(
    "m,n,compute_units,eligible",
    [
        (64, 1024, 80, False),
        (192, 128, 80, False),
        (192, 2560, 80, True),
        (193, 2560, 80, True),
        (193, 2568, 80, False),
        (193, 2568, 128, True),
    ],
)
def test_shared_row_major_shortlist_uses_first_cta_wave(
    router, m, n, compute_units, eligible
):
    configs = router._shared_row_major_candidates(m, n, compute_units)
    assert bool(configs) == eligible
    for config in configs:
        base = router._SHARED_PREFETCH_CONFIG
        assert base in router.shared_mm_configs()
        assert config == router.replace(base, group_m=1)
        for transposed in (False, True):
            options = dict(router._shared_compile_parameters(config, transposed))
            assert options["GROUP_M"] == 1
            assert options["B_K_CONTIGUOUS"] == transposed


@pytest.mark.parametrize(
    "m,n,k,compute_units,layout,eligible",
    [
        (15, 16384, 128, 64, "nn", False),
        (16, 16384, 128, 64, "nn", True),
        (64, 16384, 128, 64, "nn", True),
        (65, 16384, 128, 64, "nn", False),
        (32, 16128, 128, 64, "nn", False),
        (32, 16129, 128, 64, "nn", True),
        (32, 16384, 120, 64, "nn", False),
        (32, 16384, 128, 64, "nt", False),
        (32, 16384, 128, 80, "nn", False),
        (32, 24576, 128, 96, "nn", True),
    ],
)
def test_shared_horizontal_shortlist_uses_column_parallelism(
    router, m, n, k, compute_units, layout, eligible
):
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 0x400000)
    configs = router._shared_wide_n_candidates(m, b, compute_units)
    assert bool(configs) == eligible
    for config in configs:
        assert config.warps == (1, 4) and config.group_m == 1
        options = dict(router._shared_compile_parameters(config, False))
        assert options["num_warps"] == 4 and options["num_stages"] == 2


@pytest.mark.parametrize(
    "tile,prefetch,valid",
    [
        ((64, 256, 32), True, True),
        ((256, 256, 32), True, True),
        ((256, 256, 64), True, False),
        ((256, 256, 64), False, True),
        ((256, 256, 128), False, False),
    ],
)
def test_shared_operand_planes_fit_lds(router, tile, prefetch, valid):
    if valid:
        router.SharedMMConfig(tile, local_prefetch=prefetch)
    else:
        with pytest.raises(ValueError, match="shared memory capacity"):
            router.SharedMMConfig(tile, local_prefetch=prefetch)


@pytest.mark.parametrize(
    "supported",
    [
        set(),
        {"buffer_store_factor_offsets"},
        {"pack_bf16_rounds"},
        {"buffer_store_factor_offsets", "pack_bf16_rounds"},
    ],
)
def test_shared_paired_store_options_require_backend_support(
    router, monkeypatch, supported
):
    monkeypatch.setattr(
        router, "supports_hcu_options", lambda options: set(options) <= supported
    )
    base = router.shared_mm_configs()[0]
    paired = router.replace(base, pair_columns=True)
    tuning_options = {"buffer_store_factor_offsets", "pack_bf16_rounds"}
    compile_options = router._shared_compile_parameters.__wrapped__
    for transposed in (False, True):
        for config, expected in (
            (base, set()),
            (paired, supported),
            (router._SHARED_PACKED_A_CONFIG, set()),
            (router._SHARED_TALL_PACKED_A_CONFIG, supported),
        ):
            options = dict(compile_options(config, transposed))
            assert set(options) & tuning_options == expected


@pytest.mark.parametrize(
    "layout,supported,include_base,expected",
    [
        ("nn", True, True, True),
        ("nt", True, True, False),
        ("nn", False, True, False),
        ("nn", True, False, False),
    ],
)
def test_shared_paired_columns_reuse_short_reduction_candidate(
    router, monkeypatch, layout, supported, include_base, expected
):
    monkeypatch.setattr(router, "supports_hcu_options", lambda options: supported)
    b = tensor((512, 2048), (2048, 1) if layout == "nn" else (1, 512), 0x400000)
    base = router._SHARED_SHORT_REDUCTION_CONFIGS[0]
    configs = router._shared_paired_column_candidates(
        (base,) if include_base else (), b, m=128, compute_units=80
    )
    assert bool(configs) == expected
    assert "mmac_pair_columns" not in dict(
        router._shared_compile_parameters(base, False)
    )
    for config in configs:
        assert config.tile == base.tile and config.num_stages == base.num_stages
        assert dict(router._shared_compile_parameters(config, False))[
            "mmac_pair_columns"
        ]


@pytest.mark.parametrize(
    "m,n,units,layout,include_base,expected",
    [
        (127, 20480, 80, "nn", True, False),
        (128, 8192, 80, "nn", True, False),
        (128, 10240, 80, "nn", True, True),
        (256, 10240, 80, "nn", True, True),
        (257, 5120, 80, "nn", True, True),
        (257, 5120, 96, "nn", True, False),
        (128, 10240, 80, "nt", True, False),
        (128, 10240, 80, "nn", False, False),
    ],
)
def test_shared_wide_pair_requires_base_and_cta_coverage(
    router, monkeypatch, m, n, units, layout, include_base, expected
):
    monkeypatch.setattr(router, "supports_hcu_options", lambda _: True)
    b = tensor((1024, n), (n, 1) if layout == "nn" else (1, 1024), 0x400000)
    base = (
        router._SHARED_WIDE_FOLD_ROW_CONFIG
        if m <= 256
        else router._SHARED_WIDE_FOLD_CONFIG
    )
    configs = router._shared_paired_column_candidates(
        (base,) if include_base else (), b, m=m, compute_units=units
    )
    assert bool(configs) == expected
    assert all(config == router.replace(base, pair_columns=True) for config in configs)


@pytest.mark.parametrize("m", [512, 513])
@pytest.mark.parametrize("n", [5120, 5128])
def test_shared_row_major_pairs_require_full_column_tiles(router, monkeypatch, m, n):
    monkeypatch.setattr(router, "supports_hcu_options", lambda _: True)
    b = tensor((1024, n), (n, 1), 0x400000)
    base = router._SHARED_WIDE_FOLD_CONFIG
    fold = router._SHARED_WIDE_FOLD_ROW_CONFIG
    scalar = router.replace(fold, address_mode="scalar")

    configs = router._shared_paired_column_candidates(
        (base, fold, scalar), b, m=m, compute_units=80
    )

    allowed = (base, fold, scalar) if n == 5120 else (base,)
    assert configs == tuple(
        router.replace(config, pair_columns=True) for config in allowed
    )


@pytest.mark.parametrize("m,expected_groups", [(1280, [8]), (1281, [8, 5])])
def test_shared_pairs_add_smaller_group_after_five_row_tiles(
    router, monkeypatch, m, expected_groups
):
    monkeypatch.setattr(router, "supports_hcu_options", lambda _: True)
    b = tensor((1024, 2048), (2048, 1), 0x400000)
    configs = router._shared_paired_column_candidates(
        (router._SHARED_WIDE_FOLD_CONFIG,), b, m=m, compute_units=80
    )
    assert [config.group_m for config in configs] == expected_groups


@pytest.mark.parametrize(
    "m,n,k,units,layout,expected",
    [
        (248, 7168, 2304, 80, "nn", True),
        (256, 7040, 2304, 80, "nn", True),
        (257, 7168, 2304, 80, "nn", False),
        (257, 7168, 2304, 96, "nn", False),
        (288, 7168, 2304, 96, "nn", True),
        (256, 5120, 768, 80, "nn", True),
        (256, 4864, 768, 80, "nn", False),
        (256, 7040, 2312, 80, "nn", False),
        (256, 7040, 2304, 80, "nt", False),
    ],
)
def test_shared_four_wave_wide_pair_fits_first_cta_wave(
    router, monkeypatch, m, n, k, units, layout, expected
):
    monkeypatch.setattr(router, "supports_hcu_options", lambda _: True)
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 0x400000)
    base = router._SHARED_WIDE_PREFETCH_CONFIG
    configs = router._shared_paired_column_candidates(
        (base,), b, m=m, compute_units=units
    )
    assert configs == (
        (router.replace(base, group_m=1, pair_columns=True),) if expected else ()
    )


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("group_m", [0, 4])
@pytest.mark.parametrize(
    "shape,expected",
    [
        ((65536, 32768, 512), True),  # 4 GiB output
        ((257, 262152, 4096), True),  # B crosses 2 GiB
        ((129, 8454664, 64), True),  # wide local byte offsets and edge tiles
        ((2**31 + 1, 256, 64), False),  # block-pointer row coordinate
        ((256, 2**31 + 8, 64), False),  # producer column coordinate
        ((2**24, 2**23, 64), False),  # signed launcher grid product
        ((256 * ((2**32 - 1) // (12 * 64)) + 1, 512, 64), False),  # HIP grid X
        ((256, 256, 8192), False),  # existing pipeline limit
    ],
)
def test_cooperative_wide_address_admission(
    router, monkeypatch, shape, expected, group_m, layout
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    m, n, k = shape
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    assert router.CooperativeMMConfig(group_m).supports(a, b, c) is expected


@pytest.mark.parametrize(
    "shape,layout,expected",
    [
        ((65536, 32768, 512), "nn", True),
        ((65536, 32768, 512), "nt", True),
        ((257, 262152, 4096), "nn", True),
        ((256, 256, 2**31 - 8), "nt", False),
        ((256, 2**26, 64), "nn", False),
        ((2**30, 256, 64), "nn", False),
    ],
)
def test_shared_wide_output_preserves_block_pointer_bounds(
    router, shape, layout, expected
):
    m, n, k = shape
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    assert router.SharedMMConfig((64, 128, 32)).supports(a, b, c) is expected


@pytest.mark.parametrize(
    "m,n,k,units,fold_supported,expected_rows",
    [
        (168, 7168, 2304, 80, True, 128),
        (400, 7168, 2304, 80, True, 256),
        (400, 7168, 2304, 80, False, None),
        (256, 1792, 2304, 80, True, None),
        (256, 7168, 2304, 55, True, 256),
        (256, 7168, 2304, 27, True, None),
        (256, 7168, 2328, 80, True, None),
    ],
)
def test_packed_a_uses_shared_grid_and_amortization(
    router, monkeypatch, m, n, k, units, fold_supported, expected_rows
):
    monkeypatch.setattr(router, "supports_hcu_options", lambda _: True)
    monkeypatch.setattr(router, "_shared_address_supported", lambda: fold_supported)
    configs = router._shared_packed_a_candidates(m, n, k, units)
    if expected_rows is not None:
        (config,) = configs
        assert config.tile[0] == expected_rows and config.a_row_padding > 0
        a = tensor((m, k), (k, 1), 0x100000)
        c = tensor((m, n), (n, 1), 1 << 50)
        for stride in ((n, 1), (1, k)):
            b = tensor((k, n), stride, 1 << 49)
            assert config.supports(a, b, c)
    else:
        assert configs == ()


def test_packed_a_workspace_uses_bounded_copy_offsets(router):
    m, n, k = 65536, 256, 16384
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    config = router._SHARED_PACKED_A_CONFIG
    assert not config.supports(a, b, c)
    assert router._SHARED_WIDE_PREFETCH_CONFIG.supports(a, b, c)


@pytest.mark.parametrize("padding", [-1, True, 1.5])
def test_cooperative_padding_rejects_invalid_storage_contract(router, padding):
    with pytest.raises(ValueError, match="nonnegative integer"):
        router.CooperativeMMConfig(a_row_padding=padding)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize(
    "n,expect_padding", [(1792, False), (1800, True), (2048, True), (2056, True)]
)
def test_cooperative_padding_inherits_real_base_candidates(
    router, monkeypatch, layout, n, expect_padding
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=80),
    )
    m, k = 2048, 2304
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if layout == "nt" else (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    base = router._cooperative_mm_base_candidates(a, b, c, allow_nn=True)
    assert base
    assert router._cooperative_mm_candidates(a, b, c, allow_nn=True) == base
    expanded = router._cooperative_mm_candidates(
        a, b, c, allow_nn=True, allow_packed=True
    )
    assert expanded[: len(base)] == base
    added = expanded[len(base) :]
    assert bool(added) == expect_padding
    if added:
        assert {config.a_row_padding for config in added} == {32, 64}
        assert all(router.replace(config, a_row_padding=0) in base for config in added)
        assert all(config.supports(a, b, c) for config in added)
    if layout == "nn":
        assert {config.group_m for config in expanded} == {0, 4}


@pytest.mark.parametrize("layout", ["nn", "nt"])
def test_cooperative_padding_large_a_preserves_candidate_comparison(
    router, monkeypatch, layout
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=80),
    )
    m, n, k = 8192, 2048, 2304
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if layout == "nt" else (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    configs = router._cooperative_mm_candidates(
        a, b, c, allow_nn=True, allow_packed=True
    )
    assert {config.a_row_padding for config in configs} == {0, 32, 64}


@pytest.mark.parametrize("layout", ["nn", "nt"])
def test_cooperative_padding_checks_each_descriptor_stride(router, monkeypatch, layout):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=80),
    )
    m, n, k = 2048, 2048, 8128
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if layout == "nt" else (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    configs = router._cooperative_mm_candidates(
        a, b, c, allow_nn=True, allow_packed=True
    )
    assert {config.a_row_padding for config in configs} == {0, 32}
    assert not router.CooperativeMMConfig(4, a_row_padding=64).supports(a, b, c)


@pytest.mark.parametrize("layout", ["nn", "nt"])
def test_cooperative_padding_copy_limit_retains_wide_incumbents(
    router, monkeypatch, layout
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=80),
    )
    m, n, k = 500000, 1000000, 2304
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if layout == "nt" else (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    base = router._cooperative_mm_base_candidates(a, b, c, allow_nn=True)
    assert base and all(config.supports(a, b, c) for config in base)
    assert (
        router._cooperative_mm_candidates(a, b, c, allow_nn=True, allow_packed=True)
        == base
    )


@pytest.mark.parametrize("cu,expected", [(20, 0), (40, 1), (80, 3), (160, 4)])
def test_grouped_mv_budget_uses_device_parallelism(router, cu, expected):
    a = tensor((32, 4096), (4096, 1), 0x100000)
    b = tensor((4096, 16), (1, 4096), 0x200000000)
    configs = router.grouped_mv_candidates(a, b, compute_units=cu)
    assert len(configs) == expected
    assert all(
        32 * ((16 + config.block_n - 1) // config.block_n) <= 4 * cu
        for config in configs
    )


@pytest.mark.parametrize(
    "k,n,nt,expected",
    [
        (7168, 2112, True, True),
        (7168, 2112, False, True),
        (2048, 384, False, True),
        (2048, 384, True, False),
        (2048, 4096, False, False),
        (8192, 8192, True, False),
        (12288, 1024, True, False),
        (384, 256, True, True),
        (520, 1024, True, False),
    ],
)
def test_record_policy_candidates_use_pitch_equivalence(
    router, monkeypatch, k, n, nt, expected
):
    monkeypatch.setattr(router, "_shared_record_pitch_supported", lambda: True)
    b = tensor((k, n), (1, k) if nt else (n, 1), 0x100000000)
    configs = (
        router.SharedMMConfig((128, 128, 32)),
        router.SharedMMConfig((64, 64, 64), address_mode="scalar"),
    )
    added = router._shared_record_pitch_candidates(configs + configs, b)
    assert (
        added
        == tuple(
            router.replace(c, normalize_buffer_record_pitch=False) for c in configs
        )
        if expected
        else not added
    )
    assert configs[0].normalize_buffer_record_pitch


def test_record_policy_old_compiler_and_manual_fallback(router, monkeypatch):
    monkeypatch.setattr(router, "_shared_record_pitch_supported", lambda: False)
    config = router.SharedMMConfig((128, 128, 32))
    b = tensor((7168, 2112), (1, 7168), 0x100000000)
    assert not router._shared_record_pitch_candidates((config,), b)
    options = dict(router._shared_compile_parameters.__wrapped__(config, True))
    assert "normalize_buffer_record_pitch" not in options
    with pytest.raises(ValueError, match="compiler does not support"):
        router._shared_compile_parameters.__wrapped__(
            router.replace(config, normalize_buffer_record_pitch=False), True
        )


def test_record_policy_compile_options_and_immutable_cache(router, monkeypatch):
    monkeypatch.setattr(router, "supports_hcu_options", lambda options: True)
    config = router.SharedMMConfig((128, 128, 32))
    physical = router.replace(config, normalize_buffer_record_pitch=False)
    for nt in (False, True):
        normal = dict(router._shared_compile_parameters.__wrapped__(config, nt))
        alternate = dict(router._shared_compile_parameters.__wrapped__(physical, nt))
        assert alternate.pop("normalize_buffer_record_pitch") is False
        assert normal == alternate
    assert hash(config) != hash(physical)


@pytest.mark.parametrize("value", [0, 1, None, "physical"])
def test_record_policy_config_requires_boolean(router, value):
    with pytest.raises(ValueError, match="record pitch policy must be boolean"):
        router.SharedMMConfig((128, 128, 32), normalize_buffer_record_pitch=value)


def test_record_policy_packed_stride_and_tile_width(router, monkeypatch):
    monkeypatch.setattr(router, "_shared_record_pitch_supported", lambda: True)
    b = tensor((512, 1024), (1, 512), 0x100000000)
    base = router.SharedMMConfig((128, 128, 32))
    packed = router.replace(base, a_row_padding=128)
    assert router._shared_record_pitch_candidates((base, packed), b) == (
        router.replace(packed, normalize_buffer_record_pitch=False),
    )
    too_wide = router.SharedMMConfig((64, 512, 16))
    b = tensor((2048, 384), (384, 1), 0x100000000)
    assert not router._shared_record_pitch_candidates((too_wide,), b)


@pytest.mark.parametrize(
    "mode_name,expected",
    [("DEFAULT", False), ("EXPANDED", True), ("COST_MODEL", False)],
)
def test_splitk_family_receives_effective_expanded_mode(
    router, monkeypatch, mode_name, expected
):
    tuner = SimpleNamespace(
        _run_mode="normal",
        _flagtune_mode=getattr(router.runtime.TuningMode, mode_name),
        config_table_name="host-test",
        _benchmark_protocol=("replay", 5, 10),
    )
    monkeypatch.setattr(
        router,
        "_mm_splitk_partial",
        SimpleNamespace(fn=tuner, _apply_flagtune=lambda: None),
    )
    monkeypatch.setattr(router, "_splitk_eligible", lambda *args: True)
    received = []
    monkeypatch.setattr(
        router, "try_mm_family", lambda *args, **kwargs: received.append(kwargs) or True
    )
    assert router.try_mm_splitk(*operands(), compare_local_split=True)
    assert len(received) == 1 and received[0]["splitk_pool"] is True
    assert received[0]["expanded"] is expected


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize(
    "m,n,units,expected",
    [
        (64, 512, 80, 0),
        (65, 512, 80, 1),
        (128, 512, 80, 1),
        (65, 512, 128, 0),
        (97, 512, 128, 1),
        (128, 256, 80, 0),
        (129, 256, 80, 1),
        (256, 256, 80, 1),
    ],
)
def test_splitk_record_policy_preserves_cta_budget_and_layout_symmetry(
    router, monkeypatch, layout, m, n, units, expected
):
    k = 7168
    driver = SimpleNamespace(
        get_current_target=lambda: SimpleNamespace(backend="hip", arch="gfx936")
    )
    monkeypatch.setattr(router.triton.runtime.driver, "_active", driver)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=units),
    )
    monkeypatch.setattr(router, "_shared_supported", lambda: True)
    monkeypatch.setattr(router, "_shared_record_pitch_supported", lambda: True)
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 0x100000000)
    c = tensor((m, n), (n, 1), 0x200000000)
    default = router.mm_family_candidates(a, b, c, splitk_pool=True, expanded=False)
    expanded = router.mm_family_candidates(a, b, c, splitk_pool=True, expanded=True)
    assert expanded[: len(default)] == default
    added = expanded[len(default) :]
    assert len(added) == expected
    assert all(
        isinstance(cfg, router.SharedMMConfig) and not cfg.normalize_buffer_record_pitch
        for cfg in added
    )
    assert not any(
        isinstance(cfg, router.SharedMMConfig) and not cfg.normalize_buffer_record_pitch
        for cfg in default
    )


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize(
    "m,n,nn_admitted",
    [
        (8191, 192, False),
        (8192, 192, True),
        (8193, 192, False),
        (16384, 128, True),
        (16384, 184, False),
    ],
)
def test_cooperative_rectangular_tile_coverage_preserves_incumbents(
    router, monkeypatch, layout, m, n, nn_admitted
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=80),
    )
    k = 4096
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 0x100000000)
    c = tensor((m, n), (n, 1), 0x200000000)
    options = dict(allow_nn=True, allow_direct_tail=True, allow_packed=True)
    incumbent = router._cooperative_mm_candidates(a, b, c, **options)
    expanded = router._cooperative_mm_candidates(
        a, b, c, allow_column_tiles=True, **options
    )
    assert all(cfg.column_tile == 256 for cfg in incumbent)
    assert tuple(cfg for cfg in expanded if cfg.column_tile == 256) == incumbent
    narrow = tuple(cfg for cfg in expanded if cfg.column_tile == 128)
    # Crossing a row-tile boundary can lower useful coverage below 75%.
    # NT retains its existing tail-capable admission policy.
    assert bool(narrow) == (layout == "nt" or nn_admitted)


@pytest.mark.parametrize("units,admitted", [(40, True), (80, True), (160, False)])
def test_cooperative_rectangular_grid_scales_with_device_parallelism(
    router, monkeypatch, units, admitted
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=units),
    )
    m, n, k = 8192, 192, 4096
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1), 0x100000000)
    c = tensor((m, n), (n, 1), 0x200000000)
    configs = router._cooperative_mm_candidates(
        a, b, c, allow_nn=True, allow_column_tiles=True
    )
    assert bool(configs) == admitted


@pytest.fixture
def cooperative_columns_router(router, monkeypatch):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: True)
    monkeypatch.setattr(
        router.torch_device_fn,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=80),
    )
    monkeypatch.setattr(
        router.triton.runtime.driver,
        "_active",
        SimpleNamespace(
            get_current_target=lambda: SimpleNamespace(backend="hip", arch="gfx936")
        ),
    )
    return router


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("budget", [0, 33024])
@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("width", [128, 256])
def test_cooperative_column_tile_launch_contract(
    cooperative_columns_router, monkeypatch, layout, width, balanced, budget
):
    router = cooperative_columns_router
    m, n, k = 1025, 384, 512
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if layout == "nt" else (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    launched = []

    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launched.append((grid, args, kwargs))
                return ("artifact",)

            return run

    monkeypatch.setattr(router, "_mm_cooperative_kernel", Kernel())
    config = router.CooperativeMMConfig(
        4, column_tile=width, balanced_a_prefetch=balanced, min_shared_memory=budget
    )
    baseline_options = dict(router._COOPERATIVE_HCU_OPTIONS)
    assert router.launch_cooperative_mm(a, b, c, config) == ("artifact",)
    assert router._COOPERATIVE_HCU_OPTIONS == baseline_options
    grid, args, kwargs = launched[0]
    assert grid == (5, 3 if width == 128 else 2)
    assert all(x is y for x, y in zip(args[:3], (a, b, c)))
    assert kwargs["BN"] == width
    assert kwargs["B_K_CONTIGUOUS"] is (layout == "nt")
    assert kwargs["BALANCED_A_PREFETCH"] is balanced
    assert kwargs.get("min_shared_memory", 0) == (
        budget if budget > config.b_pool_bytes else 0
    )


@pytest.mark.parametrize("width", [64, 512, -128, 0, True, 128.0])
def test_cooperative_column_tile_invalid_contract(router, width):
    with pytest.raises(ValueError, match="column tile"):
        router.CooperativeMMConfig(column_tile=width)


@pytest.mark.parametrize("n", [128, 384])
def test_cooperative_column_tile_direct_store_is_checked_per_width(
    cooperative_columns_router, n
):
    router = cooperative_columns_router
    m, k = 1024, 512
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    assert router.CooperativeMMConfig(4, direct_tail_store=True).supports(a, b, c)
    assert not router.CooperativeMMConfig(
        4, direct_tail_store=True, column_tile=128
    ).supports(a, b, c)
    assert router.CooperativeMMConfig(4, column_tile=128).supports(a, b, c)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("n", [320, 384, 2048, 2056])
def test_cooperative_column_tile_candidates_preserve_incumbents(
    cooperative_columns_router, layout, n
):
    router = cooperative_columns_router
    m, k = 16384, 2304
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if layout == "nt" else (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    options = dict(allow_nn=True, allow_direct_tail=True, allow_packed=True)
    old = router._cooperative_mm_candidates(a, b, c, **options)
    new = router._cooperative_mm_candidates(a, b, c, allow_column_tiles=True, **options)
    assert new[: len(old)] == old
    assert all(config.column_tile == 128 for config in new[len(old) :])
    if layout == "nn":
        assert {config.group_m for config in new} == {0, 4}
    default = router._cooperative_mm_candidates(a, b, c, allow_nn=False)
    assert all(config.column_tile == 256 for config in default)


@pytest.mark.parametrize("n", [128, 320])
def test_cooperative_column_tile_nn_budget_uses_rectangular_tile(
    cooperative_columns_router, n
):
    router = cooperative_columns_router
    m, k = 16384, 4096
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    assert not router._cooperative_mm_base_candidates(a, b, c, allow_nn=True)
    configs = router._cooperative_mm_base_candidates(
        a, b, c, allow_nn=True, allow_column_tiles=True
    )
    assert configs and all(
        config.column_tile == 128 and config.group_m in (0, 4) for config in configs
    )


def test_cooperative_column_tile_unsupported_path_does_not_query_gpu(
    router, monkeypatch
):
    monkeypatch.setattr(router, "_cooperative_supported", lambda: False)

    def forbidden(*args):
        raise AssertionError("unsupported candidate queried GPU properties")

    monkeypatch.setattr(router.torch_device_fn, "get_device_properties", forbidden)
    a = tensor((16384, 4096), (4096, 1), 0x100000)
    b = tensor((4096, 384), (384, 1), 1 << 49)
    c = tensor((16384, 384), (384, 1), 1 << 50)
    assert not router._cooperative_mm_candidates(
        a, b, c, allow_nn=True, allow_column_tiles=True
    )


def test_cooperative_column_tile_old_serialized_config_retains_default_grid(
    cooperative_columns_router, monkeypatch
):
    router = cooperative_columns_router
    cfg = router.CooperativeMMConfig(
        **dict(group_m=4, direct_tail_store=False, a_row_padding=0)
    )
    assert cfg.column_tile == 256
    a = tensor((1024, 512), (512, 1), 0x100000)
    b = tensor((512, 384), (1, 512), 1 << 49)
    c = tensor((1024, 384), (384, 1), 1 << 50)
    grids = []

    class Kernel:
        def __getitem__(self, grid):
            grids.append(grid)
            return lambda *a, **kw: ("artifact",)

    monkeypatch.setattr(router, "_mm_cooperative_kernel", Kernel())
    router.launch_cooperative_mm(a, b, c, cfg)
    assert grids == [(4, 2)]


@pytest.mark.parametrize(
    "width,budget,supported",
    [
        (128, 0, True),
        (128, 1, True),
        (128, 32768, True),
        (128, 33024, False),
        (256, 33024, True),
        (256, 65536, True),
    ],
)
def test_cooperative_reservation_old_compiler_uses_proved_noop_budget(
    cooperative_columns_router, monkeypatch, width, budget, supported
):
    router = cooperative_columns_router
    monkeypatch.setattr(
        router,
        "supports_hcu_options",
        lambda options: "min_shared_memory" not in options,
    )
    m, n, k = 1024, 384, 512
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append(kwargs)

            return run

    monkeypatch.setattr(router, "_mm_cooperative_kernel", Kernel())
    config = router.CooperativeMMConfig(column_tile=width, min_shared_memory=budget)
    assert config.supports(a, b, c) is supported
    if supported:
        original = dict(router._COOPERATIVE_HCU_OPTIONS)
        router.launch_cooperative_mm(a, b, c, config)
        assert len(calls) == 1 and "min_shared_memory" not in calls[0]
        assert router._COOPERATIVE_HCU_OPTIONS == original
    else:
        with pytest.raises(ValueError, match="unsupported"):
            router.launch_cooperative_mm(a, b, c, config)
        assert not calls


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("capability", [False, True])
def test_cooperative_reservation_is_optional_and_preserves_old_pool(
    cooperative_columns_router, monkeypatch, layout, capability
):
    router = cooperative_columns_router
    monkeypatch.setattr(
        router,
        "supports_hcu_options",
        lambda options: capability or "min_shared_memory" not in options,
    )
    m, n, k = 16384, 384, 7168
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if layout == "nt" else (n, 1), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    flags = dict(allow_nn=True, allow_column_tiles=True, allow_balanced_prefetch=True)
    original = router._cooperative_mm_candidates(a, b, c, **flags)
    expanded = router._cooperative_mm_candidates(
        a, b, c, allow_shared_reservation=True, **flags
    )
    assert expanded[: len(original)] == original
    added = expanded[len(original) :]
    assert bool(added) is capability
    assert all(cfg.min_shared_memory > cfg.b_pool_bytes for cfg in added)
    expected = tuple(cfg for cfg in original if cfg.b_pool_bytes < 33024)
    if capability:
        assert (
            tuple(router.replace(cfg, min_shared_memory=0) for cfg in added) == expected
        )
    else:
        assert not added


def test_shared_config_normalizes_immutable_sequences(router):
    config = router.SharedMMConfig([64, 128, 32], warps=[2, 2])
    canonical = router.SharedMMConfig((64, 128, 32), warps=(2, 2))
    assert config == canonical and hash(config) == hash(canonical)
