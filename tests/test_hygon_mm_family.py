"""Real family router cache, graph-capture and dynamic operand contracts."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


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
        stride=lambda: stride,
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
def test_optional_mm_compile_failure_preserves_incumbent(
    router, monkeypatch, fail_on_cache_hit, clear_on_failure
):
    from triton.runtime.errors import OutOfResources

    install_benchmark(router, monkeypatch)
    config = router.SharedMMConfig((64, 128, 32))
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
        router.torch_device_fn, "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=128),
    )
    a = tensor((m, k), (k, 1), 0x100000)
    c = tensor((m, n), (n, 1), 0x200000000)
    nn = tensor((k, n), (n, 1), 0x100000000)
    nt = tensor((k, n), (1, k), 0x100000000)
    assert router.short_k_mm_candidates(a, nn, c) == router.SHORT_K_MM_CONFIGS
    assert router.short_k_mm_candidates(a, nt, c) == router.SHORT_K_MM_CONFIGS
    assert not router._SHORT_K_STORAGE.supports(a, nn, c)


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
        knobs, "hcu",
        SimpleNamespace(
            use_buffer_ops=True, use_async_copy=False,
            use_block_pingpong=False, use_in_thread_transpose=False,
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
        (8, 64, 4096, True, 0),
        (4, 2048, 4096, True, 0),
    ],
)
def test_grouped_mv_candidates_follow_layout_and_parallelism(router, m, n, k, nt, expected):
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (1, k) if nt else (n, 1), 0x100000000)
    assert len(router.grouped_mv_candidates(a, b, compute_units=80)) == expected


@pytest.mark.parametrize(
    "m,n,expected_tiles",
    [
        (2, 64, ((16, 16), (16, 16))),
        (128, 128, ((16, 16), (16, 16), (32, 32))),
        (4, 2048, ((16, 16), (16, 16), (32, 32), (32, 64, 64))),
    ],
)
def test_splitk_shortlist_uses_tile_parallelism(router, monkeypatch, m, n, expected_tiles):
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
        router, "local_split_mm_configs", lambda: router._SPLITK_LOCAL + (router._SPLITK_LOCAL_WIDE,)
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
        (4096, 2048, 192, "nn", True, 0, (4,)),
        (3072, 2048, 192, "nn", True, 0, (4,)),
        (2816, 2048, 192, "nn", True, 0, ()),
        (16384, 376, 192, "nn", True, 0, ()),
        (16384, 392, 192, "nn", True, 0, (4,)),
        (4097, 2049, 192, "nn", True, 0, ()),
        (4097, 2056, 192, "nn", True, 2, ()),
        (4097, 2056, 193, "nn", True, 0, ()),
        (4097, 2056, 192, "nn", False, 0, ()),
        (512, 257, 192, "nt", False, 0, (0, 4)),
        (65536, 32768, 512, "nn", True, 0, (4,)),
        (512, 262152, 4096, "nn", True, 0, (4,)),
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
def test_shared_wide_output_preserves_block_pointer_bounds(router, shape, layout, expected):
    m, n, k = shape
    a = tensor((m, k), (k, 1), 0x100000)
    b = tensor((k, n), (n, 1) if layout == "nn" else (1, k), 1 << 49)
    c = tensor((m, n), (n, 1), 1 << 50)
    assert router.SharedMMConfig((64, 128, 32)).supports(a, b, c) is expected
