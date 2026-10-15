import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

pytestmark = pytest.mark.nested_view_from_buffer_copy


@pytest.mark.nested_view_from_buffer_copy
@pytest.mark.parametrize("metadata_placement", ["host", "device", "mixed"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_view_from_buffer_copy(dtype, metadata_placement):
    buffer_size = 100000
    buffer = torch.randn(buffer_size, dtype=dtype, device=flag_gems.device)
    sizes_data = [[1000], [2000], [3000]]
    strides_data = [[1], [1], [1]]
    offsets_data = [0, 1000, 3000]
    metadata_device = flag_gems.device

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

    ref_out_cpu = torch.ops.aten._nested_view_from_buffer_copy.default(
        buffer.cpu(),
        sizes.cpu(),
        strides.cpu(),
        offsets.cpu(),
    )
    res_out = flag_gems._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)

    assert res_out.is_nested
    assert res_out.device == buffer.device
    assert ref_out_cpu.is_nested

    res_unbind = res_out.unbind()
    ref_unbind = ref_out_cpu.unbind()
    assert len(res_unbind) == len(ref_unbind)
    for res_t, ref_t in zip(res_unbind, ref_unbind):
        assert res_t.shape == ref_t.shape
        ref_t_matched = ref_t if utils.TO_CPU else ref_t.to(res_t.device)
        utils.gems_assert_close(res_t, ref_t_matched, dtype)


@pytest.mark.nested_view_from_buffer_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_view_from_buffer_copy_oversized_tail(dtype):
    buffer = torch.randn(100000, dtype=dtype, device=flag_gems.device)
    sizes = torch.tensor(
        [[1000], [2000], [3000]], dtype=torch.int64, device=flag_gems.device
    )
    strides = torch.ones((3, 1), dtype=torch.int64, device=flag_gems.device)
    offsets = torch.tensor([0, 1000, 3000], dtype=torch.int64, device=flag_gems.device)

    ref_out_cpu = torch.ops.aten._nested_view_from_buffer_copy.default(
        buffer.cpu(), sizes.cpu(), strides.cpu(), offsets.cpu()
    )
    res_out = flag_gems._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)

    assert res_out.is_nested
    assert res_out.device == buffer.device
    for res_t, ref_t in zip(res_out.unbind(), ref_out_cpu.unbind()):
        ref_t_matched = ref_t if utils.TO_CPU else ref_t.to(res_t.device)
        utils.gems_assert_close(res_t, ref_t_matched, dtype)
