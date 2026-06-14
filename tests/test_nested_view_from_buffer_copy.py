import sys

# The concurrent agent runner may leave a stale editable install redirector pointing to
# a different worktree. Remove it so our worktree's flag_gems is loaded.
sys.meta_path = [
    m for m in sys.meta_path if "ScikitBuildRedirectingFinder" not in str(type(m))
]

# Swap our worktree's src to the front of sys.path.
_our_src = "/tmp/flaggems_agent_worktrees/agent__nested_view_from_buffer_copy_179605-1781252494/src"
sys.path = [_our_src] + [p for p in sys.path if p != _our_src]

# Clear any cached flag_gems modules.
for _k in list(sys.modules):
    if "flag_gems" in _k:
        del sys.modules[_k]

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

pytestmark = pytest.mark.nested_view_from_buffer_copy


@pytest.mark.nested_view_from_buffer_copy
# _nested_view_from_buffer_copy only supports float16 and float32 on Metax
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nested_view_from_buffer_copy(dtype):
    # Create buffer tensor
    buffer_size = 100000
    buffer = torch.randn(buffer_size, dtype=dtype, device=flag_gems.device)

    # Define nested tensor parameters
    sizes = torch.tensor(
        [[1000], [2000], [3000]], dtype=torch.int64, device=flag_gems.device
    )
    strides = torch.tensor([[1], [1], [1]], dtype=torch.int64, device=flag_gems.device)
    offsets = torch.tensor([0, 1000, 3000], dtype=torch.int64, device=flag_gems.device)

    ref_buffer = utils.to_reference(buffer)
    ref_sizes = utils.to_reference(sizes)
    ref_strides = utils.to_reference(strides)
    ref_offsets = utils.to_reference(offsets)

    # torch.ops.aten._nested_view_from_buffer_copy segfaults on CUDA (Metax bug);
    # must run reference on CPU explicitly.
    ref_out_cpu = torch.ops.aten._nested_view_from_buffer_copy.default(
        ref_buffer.to("cpu"),
        ref_sizes.to("cpu"),
        ref_strides.to("cpu"),
        ref_offsets.to("cpu"),
    )

    with flag_gems.use_gems():
        res_out = flag_gems._nested_view_from_buffer_copy(
            buffer, sizes, strides, offsets
        )

    # Verify the nested tensor structure matches
    assert res_out.is_nested
    assert ref_out_cpu.is_nested

    # Verify each component tensor has correct properties
    res_unbind = torch.unbind(res_out)
    ref_unbind = torch.unbind(ref_out_cpu)

    assert len(res_unbind) == len(ref_unbind)
    for i, (res_t, ref_t) in enumerate(zip(res_unbind, ref_unbind)):
        assert res_t.shape == ref_t.shape
        # In quick-cpu mode, gems_assert_close handles CPU conversion internally
        # and requires ref on CPU. In GPU mode, ref must be on the same device.
        ref_t_matched = ref_t if utils.TO_CPU else ref_t.to(res_t.device)
        utils.gems_assert_close(res_t, ref_t_matched, dtype)
