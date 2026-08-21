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

import builtins
import gc
import importlib.util
import weakref
from pathlib import Path

import pytest
import torch

_BACKEND_INDEX_ADD_PATHS = {
    "mthreads": Path("src/flag_gems/runtime/backend/_mthreads/ops/index_add.py"),
    "metax": Path("src/flag_gems/runtime/backend/_metax/ops/index_add.py"),
}


def _count_bounds_reads(monkeypatch, module):
    reads = 0
    original = module._read_index_bounds

    def counting_reader(index):
        nonlocal reads
        reads += 1
        return original(index)

    monkeypatch.setattr(module, "_read_index_bounds", counting_reader)
    return lambda: reads


def _load_backend_module(monkeypatch, backend, block_shared_cache=False):
    if block_shared_cache:
        original_import = builtins.__import__

        def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "flag_gems.utils.index_bounds_cache":
                raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

    path = Path(__file__).parents[1] / _BACKEND_INDEX_ADD_PATHS[backend]
    spec = importlib.util.spec_from_file_location(
        f"_isolated_{backend}_index_add", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=_BACKEND_INDEX_ADD_PATHS)
def cache_module(monkeypatch, request):
    return _load_backend_module(monkeypatch, request.param)


@pytest.mark.parametrize("backend", _BACKEND_INDEX_ADD_PATHS)
def test_vendor_index_add_cache_is_single_file_loadable(monkeypatch, backend):
    module = _load_backend_module(monkeypatch, backend, block_shared_cache=True)
    cache = module._IndexBoundsCache(max_entries=1, max_bytes=16)

    cache.assert_in_bounds(torch.tensor([0], dtype=torch.int64), 1)


def test_equivalent_storage_slice_views_reuse_successful_validation(
    monkeypatch, cache_module
):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    base = torch.tensor([99, 0, 2, 4, 99], dtype=torch.int64)
    first_view = base[1:4]
    second_view = base[1:4]

    assert first_view is not second_view
    cache.assert_in_bounds(first_view, 5)
    cache.assert_in_bounds(second_view, 5)

    assert read_count() == 1


def test_tensor_version_change_forces_revalidation(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    base = torch.tensor([0, 1, 2], dtype=torch.int64)

    cache.assert_in_bounds(base[:], 3)
    base[2] = 3

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        cache.assert_in_bounds(base[:], 3)
    assert read_count() == 2


def test_upper_bound_is_part_of_cache_identity(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    index = torch.tensor([0, 3], dtype=torch.int64)

    cache.assert_in_bounds(index, 4)

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        cache.assert_in_bounds(index, 3)
    assert read_count() == 2


def test_view_metadata_is_part_of_cache_identity(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    base = torch.tensor([0, 1, 4], dtype=torch.int64)

    cache.assert_in_bounds(base[:2], 4)

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        cache.assert_in_bounds(base[:3], 4)
    assert read_count() == 2


def test_independent_alias_with_same_pointer_and_version_does_not_hit(
    monkeypatch, cache_module
):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    index = torch.tensor([0, 1], dtype=torch.int64)
    independent_alias = index.data

    assert independent_alias is not index
    assert independent_alias.data_ptr() == index.data_ptr()
    assert independent_alias._version == index._version
    assert independent_alias._base is None
    cache.assert_in_bounds(index, 2)
    cache.assert_in_bounds(independent_alias, 2)

    assert read_count() == 2


def test_normal_view_cache_does_not_hide_lazy_negative_oob(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    normal = torch.tensor([0, 1], dtype=torch.int64)
    lazy_negative = torch._neg_view(normal)

    assert lazy_negative.is_neg()
    assert lazy_negative.data_ptr() == normal.data_ptr()
    assert lazy_negative._version == normal._version
    cache.assert_in_bounds(normal, 2)

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        cache.assert_in_bounds(lazy_negative, 2)
    assert read_count() == 2


def test_lazy_negative_validation_is_not_reused_by_normal_view(
    monkeypatch, cache_module
):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    normal = torch.tensor([-1, 0], dtype=torch.int64)
    lazy_negative = torch._neg_view(normal)

    cache.assert_in_bounds(lazy_negative, 2)

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        cache.assert_in_bounds(normal, 2)
    assert read_count() == 2


def test_lazy_conjugate_tensor_is_never_cached(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    reads = 0

    def readable_complex_bounds(index):
        nonlocal reads
        reads += 1
        return 0, 1

    monkeypatch.setattr(module, "_read_index_bounds", readable_complex_bounds)
    lazy_conjugate = torch.tensor([0j, 1j]).conj()
    assert lazy_conjugate.is_conj()

    cache.assert_in_bounds(lazy_conjugate, 2)
    cache.assert_in_bounds(lazy_conjugate, 2)

    assert reads == 2


def test_kernel_index_resolver_materializes_lazy_negative_values(cache_module):
    normal = torch.tensor([1, 0], dtype=torch.int64)
    assert cache_module._resolve_index_for_kernel(normal) is normal

    raw = torch.tensor([-1, 0], dtype=torch.int64)
    lazy_negative = torch._neg_view(raw)
    resolved = cache_module._resolve_index_for_kernel(lazy_negative)

    assert lazy_negative.is_neg()
    assert not resolved.is_neg()
    assert resolved.tolist() == [1, 0]
    assert resolved.data_ptr() != lazy_negative.data_ptr()


def test_kernel_index_resolver_bypasses_registered_resolve_neg(
    monkeypatch, cache_module
):
    raw = torch.tensor([-1, 0], dtype=torch.int64)
    lazy_negative = torch._neg_view(raw)
    original_neg = torch.neg
    neg_input_bits = []

    def broken_registered_resolve_neg(self):
        return torch._neg_view(self)

    def recording_neg(value):
        neg_input_bits.append(value.is_neg())
        return original_neg(value)

    monkeypatch.setattr(torch.Tensor, "resolve_neg", broken_registered_resolve_neg)
    monkeypatch.setattr(torch, "neg", recording_neg)
    resolved = cache_module._resolve_index_for_kernel(lazy_negative)

    assert neg_input_bits == [False]
    assert not resolved.is_neg()
    assert resolved.tolist() == [1, 0]


def test_cacheable_false_forces_revalidation(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    index = torch.tensor([0, 1], dtype=torch.int64)

    cache.assert_in_bounds(index, 2, cacheable=False)
    cache.assert_in_bounds(index, 2, cacheable=False)

    assert read_count() == 2


def test_hit_rechecks_identity_after_initial_key_read(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    index = torch.tensor([0, 1], dtype=torch.int64)
    cache.assert_in_bounds(index, 2)
    original_identity = module._index_bounds_cache_identity
    mutated = False

    def mutate_after_identity_read(tensor, upper_bound):
        nonlocal mutated
        identity = original_identity(tensor, upper_bound)
        if not mutated:
            tensor[1] = upper_bound
            mutated = True
        return identity

    monkeypatch.setattr(
        module, "_index_bounds_cache_identity", mutate_after_identity_read
    )

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        cache.assert_in_bounds(index, 2)
    assert read_count() == 2


def test_failed_validation_is_not_cached(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    index = torch.tensor([-1, 0], dtype=torch.int64)

    for _ in range(2):
        with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
            cache.assert_in_bounds(index, 4)

    assert read_count() == 2


def test_inference_tensor_without_version_is_never_cached(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    with torch.inference_mode():
        index = torch.tensor([0, 1], dtype=torch.int64)

    cache.assert_in_bounds(index, 2)
    cache.assert_in_bounds(index, 2)

    assert read_count() == 2


def test_cache_evicts_least_recently_used_entry(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=2)
    read_count = _count_bounds_reads(monkeypatch, module)
    first = torch.tensor([0], dtype=torch.int64)
    second = torch.tensor([1], dtype=torch.int64)
    third = torch.tensor([2], dtype=torch.int64)

    cache.assert_in_bounds(first, 3)
    cache.assert_in_bounds(second, 3)
    cache.assert_in_bounds(first, 3)
    cache.assert_in_bounds(third, 3)
    cache.assert_in_bounds(second, 3)

    assert read_count() == 4


def test_cache_does_not_keep_validated_tensor_or_root_alive(
    cache_module,
):
    cache = cache_module._IndexBoundsCache(max_entries=1)
    root = torch.tensor([9, 0, 1, 9], dtype=torch.int64)
    index = root[1:3]
    retained_index = weakref.ref(index)
    retained_root = weakref.ref(root)

    cache.assert_in_bounds(index, 2)
    del index
    del root
    gc.collect()

    assert retained_index() is None
    assert retained_root() is None


def test_dead_root_weakref_never_satisfies_colliding_key(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    original_identity = module._index_bounds_cache_identity
    forced_key = ("forced-dead-root-collision",)

    def colliding_identity(index, upper_bound):
        identity = original_identity(index, upper_bound)
        if identity is None:
            return None
        _, version_root = identity
        return forced_key, version_root

    monkeypatch.setattr(module, "_index_bounds_cache_identity", colliding_identity)
    first = torch.tensor([0], dtype=torch.int64)
    first_ref = weakref.ref(first)
    cache.assert_in_bounds(first, 1)
    del first
    gc.collect()
    assert first_ref() is None

    invalid = torch.tensor([1], dtype=torch.int64)
    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        cache.assert_in_bounds(invalid, 1)
    assert read_count() == 2


def test_mthreads_noncontiguous_index_copy_is_not_retained(monkeypatch):
    module = _load_backend_module(monkeypatch, "mthreads")
    module._INDEX_BOUNDS_CACHE = module._IndexBoundsCache(max_entries=4)
    read_count = _count_bounds_reads(monkeypatch, module)
    storage = torch.tensor([[0, 9], [1, 9]], dtype=torch.int64)
    noncontiguous_index = storage[:, 0]
    assert not noncontiguous_index.is_contiguous()

    temporary = noncontiguous_index.contiguous()
    temporary_ref = weakref.ref(temporary)
    module._assert_index_in_bounds(temporary, 2)
    del temporary
    gc.collect()
    assert temporary_ref() is None

    storage[1, 0] = 2
    second_temporary = noncontiguous_index.contiguous()
    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        module._assert_index_in_bounds(second_temporary, 2)
    assert read_count() == 2


def test_entry_larger_than_byte_budget_is_not_cached(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4, max_bytes=8)
    read_count = _count_bounds_reads(monkeypatch, module)
    index = torch.tensor([0, 1], dtype=torch.int64)

    cache.assert_in_bounds(index, 2)
    cache.assert_in_bounds(index, 2)

    assert read_count() == 2


def test_cache_evicts_lru_entry_to_stay_within_byte_budget(monkeypatch, cache_module):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=4, max_bytes=16)
    read_count = _count_bounds_reads(monkeypatch, module)
    first = torch.tensor([0], dtype=torch.int64)
    second = torch.tensor([1], dtype=torch.int64)
    third = torch.tensor([2], dtype=torch.int64)

    cache.assert_in_bounds(first, 3)
    cache.assert_in_bounds(second, 3)
    cache.assert_in_bounds(first, 3)
    cache.assert_in_bounds(third, 3)
    cache.assert_in_bounds(second, 3)

    assert read_count() == 4


@pytest.mark.parametrize("max_entries", [0, -1])
def test_nonpositive_entry_limit_disables_cache_without_error(
    monkeypatch, max_entries, cache_module
):
    module = cache_module
    cache = module._IndexBoundsCache(max_entries=max_entries)
    read_count = _count_bounds_reads(monkeypatch, module)
    index = torch.tensor([0], dtype=torch.int64)

    cache.assert_in_bounds(index, 1)
    cache.assert_in_bounds(index, 1)

    assert read_count() == 2
