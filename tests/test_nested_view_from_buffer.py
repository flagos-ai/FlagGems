import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Component layouts: each case is (buffer_size, sizes, strides, offsets).
# The 2-D component shapes match the (num_components, dim) metadata the native
# kernel validates; buffer sizes keep the payload scale of the sibling
# _nested_view_from_buffer_copy cases so the per-case cost stays comparable.
# Case 3 carries a zero-size component, which the native kernel accepts.
VIEW_CASES = [
    (
        24000,
        [[1000, 4], [2000, 4], [3000, 4]],
        [[4, 1], [4, 1], [4, 1]],
        [0, 1000, 3000],
    ),
    # buffer_size = last_offset + prod(last sizes) = 36000 + 3000*4*3
    (
        72000,
        [[1000, 4, 3], [2000, 4, 3], [3000, 4, 3]],
        [[12, 3, 1], [12, 3, 1], [12, 3, 1]],
        [0, 12000, 36000],
    ),
    (
        16000,
        [[1000, 4], [0, 4], [3000, 4]],
        [[4, 1], [4, 1], [4, 1]],
        [0, 4000, 4000],
    ),
]


def _cpu_reference(buffer, sizes, strides, offsets):
    """Build the reference with the native op on host-resident tensors.

    The native kernel dereferences the metadata through CPU raw pointers, so
    device-resident metadata must be staged before the reference call too. The
    reference is built on the CPU in every mode: the result is a strided
    nested tensor whose per-component structure (shapes, strides, offsets) is
    device-independent, and keeping it on the host matches how the sibling
    _nested_view_from_buffer_copy test compares against a CPU reference.
    """
    # Host-resident staging is required here (see docstring): the metadata is
    # dereferenced through CPU raw pointers, so to_reference on the nested
    # result is not an option.
    return torch.ops.aten._nested_view_from_buffer.default(
        buffer.cpu(), sizes.cpu(), strides.cpu(), offsets.cpu()
    )


def _assert_nested_view_equal(res, ref, buffer, dtype):
    """Structural comparison for strided-layout nested tensors.

    ``.shape``/``.size()`` cannot be queried on a strided nested tensor on
    this torch build (NestedTensorImpl raises "doesn't support sizes"), so the
    components are compared through ``unbind``.
    """
    assert res.is_nested
    assert ref.is_nested
    assert res.layout == torch.strided
    assert res.layout == ref.layout
    # The Gems result lives on the input buffer's device, independent of where
    # the reference was built.
    assert res.device == buffer.device

    res_unbind = res.unbind()
    ref_unbind = ref.unbind()
    assert len(res_unbind) == len(ref_unbind)
    for res_t, ref_t in zip(res_unbind, ref_unbind):
        assert res_t.shape == ref_t.shape
        assert res_t.stride() == ref_t.stride()
        assert res_t.dtype == ref_t.dtype
        assert res_t.device == buffer.device
        ref_t_matched = ref_t if utils.TO_CPU else ref_t.to(res_t.device)
        utils.gems_assert_close(res_t, ref_t_matched, dtype)

    # The metadata queries must reproduce the requested sizes/strides/offsets.
    assert torch.equal(
        torch.ops.aten._nested_tensor_size.default(res).cpu(),
        torch.ops.aten._nested_tensor_size.default(ref).cpu(),
    )
    assert torch.equal(
        torch.ops.aten._nested_tensor_strides.default(res).cpu(),
        torch.ops.aten._nested_tensor_strides.default(ref).cpu(),
    )
    assert torch.equal(
        torch.ops.aten._nested_tensor_storage_offsets.default(res).cpu(),
        torch.ops.aten._nested_tensor_storage_offsets.default(ref).cpu(),
    )


@pytest.mark.nested_view_from_buffer
@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("metadata_placement", ["host", "device", "mixed"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("view_case", VIEW_CASES)
def test_nested_view_from_buffer(view_case, dtype, metadata_placement):
    buffer_size, sizes_data, strides_data, offsets_data = view_case
    buffer = torch.randn(buffer_size, dtype=dtype, device=flag_gems.device)
    metadata_device = flag_gems.device

    # "device" and "mixed" exercise the host-staging repair in the wrapper:
    # metadata arriving on the GPU would segfault the raw native kernel.
    sizes = torch.tensor(
        sizes_data,
        dtype=torch.int64,
        device=metadata_device if metadata_placement == "device" else "cpu",
    )
    strides = torch.tensor(
        strides_data,
        dtype=torch.int64,
        device=metadata_device if metadata_placement != "mixed" else "cpu",
    )
    offsets = torch.tensor(
        offsets_data,
        dtype=torch.int64,
        device=(
            metadata_device if metadata_placement in ("device", "mixed") else "cpu"
        ),
    )

    ref_out = _cpu_reference(buffer, sizes, strides, offsets)
    res_out = flag_gems._nested_view_from_buffer(buffer, sizes, strides, offsets)

    _assert_nested_view_equal(res_out, ref_out, buffer, dtype)


@pytest.mark.nested_view_from_buffer
@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_view_from_buffer_aliasing(dtype):
    # The schema annotates both self and the result with (a): the returned
    # nested tensor must alias the buffer, so writes through an unbound
    # component have to be visible in the underlying buffer. This is a pure
    # device-local property, so it holds in --ref=cpu mode as well.
    buffer = torch.randn(8000, dtype=dtype, device=flag_gems.device)
    sizes = torch.tensor([[1000, 4], [1000, 4]], dtype=torch.int64)
    strides = torch.tensor([[4, 1], [4, 1]], dtype=torch.int64)
    offsets = torch.tensor([0, 4000], dtype=torch.int64)

    res_out = flag_gems._nested_view_from_buffer(buffer, sizes, strides, offsets)
    res_unbind = res_out.unbind()

    assert all(
        t.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        for t in res_unbind
    )
    # Write a value exactly representable in every FLOAT_DTYPE (halves keep
    # full integer precision up to 2048) and check it lands in the buffer.
    res_unbind[0][0, 2] = 1024.0
    assert buffer[2].item() == 1024.0


@pytest.mark.nested_view_from_buffer
# float32 only: the case covers metadata dimension handling, not dtype
# coverage, which the parametrized test above already provides.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_view_from_buffer_3d_components(dtype):
    # A 3-D component shape (metadata dim=3) confirms the
    # (num_components, dim) handling generalizes beyond 2-D components.
    # float32 only: the case exists to cover metadata dimension handling, not
    # dtype coverage, which the parametrized test above already provides.
    buffer_size, sizes_data, strides_data, offsets_data = VIEW_CASES[1]
    buffer = torch.randn(buffer_size, dtype=dtype, device=flag_gems.device)
    sizes = torch.tensor(sizes_data, dtype=torch.int64)
    strides = torch.tensor(strides_data, dtype=torch.int64)
    offsets = torch.tensor(offsets_data, dtype=torch.int64)

    ref_out = _cpu_reference(buffer, sizes, strides, offsets)
    res_out = flag_gems._nested_view_from_buffer(buffer, sizes, strides, offsets)

    _assert_nested_view_equal(res_out, ref_out, buffer, dtype)
