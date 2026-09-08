import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

BACKOFF_FACTOR = 0.5


# ``_amp_update_scale_`` is a scalar (0-d) operation driven by the AMP loss
# scaler. PyTorch requires ``self`` (the current scale) and ``found_inf`` to be
# float32 scalars and ``growth_tracker`` to be an int32 scalar, so the dtype is
# fixed and we exercise the logic through representative input combinations.
GROWTH_FACTOR = 2.0


GROWTH_INTERVAL = 5


# (current_scale, current_tracker, found_inf) cases covering the growth,
# increment, backoff and edge-value branches.
SCALE_TRACKER_FOUNDINF_CASES = [
    # Growth branch: tracker + 1 == growth_interval -> scale grows, tracker reset.
    (1024.0, 4, 0.0),
    # Increment branch: tracker + 1 < growth_interval -> tracker increments.
    (1024.0, 0, 0.0),
    (1024.0, 2, 0.0),
    # Tracker already past interval -> just increments, no growth.
    (1024.0, 5, 0.0),
    # Backoff branch: found_inf non-zero -> scale shrinks, tracker reset.
    (1024.0, 3, 1.0),
    (1024.0, 0, 1.0),
    # Edge values for found_inf: tiny positive / negative both trigger backoff
    # (non-zero), while -0.0 does not (it equals 0).
    (1024.0, 4, 1e-20),
    (1024.0, 4, -1.0),
    (1024.0, 4, -0.0),
]


def _make_inputs(scale_val, tracker_val, found_inf_val):
    scale = torch.tensor(scale_val, device=flag_gems.device, dtype=torch.float32)
    tracker = torch.tensor(tracker_val, device=flag_gems.device, dtype=torch.int32)
    found_inf = torch.tensor(
        found_inf_val, device=flag_gems.device, dtype=torch.float32
    )
    return scale, tracker, found_inf


@pytest.mark.amp_update_scale_
@pytest.mark.parametrize(
    "scale_val,tracker_val,found_inf_val", SCALE_TRACKER_FOUNDINF_CASES
)
def test_amp_update_scale_(scale_val, tracker_val, found_inf_val):
    """Compare the GEMS implementation against PyTorch for the AMP scale update."""
    # Reference
    ref_scale = utils.to_reference(
        _make_inputs(scale_val, tracker_val, found_inf_val)[0]
    )
    ref_tracker = utils.to_reference(
        _make_inputs(scale_val, tracker_val, found_inf_val)[1]
    )
    ref_found_inf = utils.to_reference(
        _make_inputs(scale_val, tracker_val, found_inf_val)[2]
    )
    ref_out = torch._amp_update_scale_(
        ref_scale,
        ref_tracker,
        ref_found_inf,
        GROWTH_FACTOR,
        BACKOFF_FACTOR,
        GROWTH_INTERVAL,
    )

    # GEMS
    res_scale, res_tracker, res_found_inf = _make_inputs(
        scale_val, tracker_val, found_inf_val
    )
    with flag_gems.use_gems():
        res_out = torch._amp_update_scale_(
            res_scale,
            res_tracker,
            res_found_inf,
            GROWTH_FACTOR,
            BACKOFF_FACTOR,
            GROWTH_INTERVAL,
        )

    # The return value is the (in-place updated) scale tensor itself.
    assert res_out is res_scale
    utils.gems_assert_equal(res_scale, ref_scale)
    utils.gems_assert_equal(res_tracker, ref_tracker)
    # found_inf is not modified.
    utils.gems_assert_equal(res_found_inf, ref_found_inf)
    # Returned scale must match the in-place scale.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.amp_update_scale_
def test_amp_update_scale__growth_walk():
    """Walk the growth tracker over a full interval and confirm the scale grows."""
    interval = 3
    scale_val = 4.0

    def step(use_gems, scale, tracker):
        found_inf = torch.tensor(
            0.0,
            device=scale.device,
            dtype=torch.float32,
        )
        if use_gems:
            with flag_gems.use_gems():
                torch._amp_update_scale_(scale, tracker, found_inf, 2.0, 0.5, interval)
        else:
            torch._amp_update_scale_(scale, tracker, found_inf, 2.0, 0.5, interval)

    # Reference walk.  ``to_reference`` moves the scalars to the CPU reference
    # device under ``--ref=cpu`` so the subsequent ``gems_assert_equal`` ``to_cpu``
    # contract (``ref`` already on CPU) holds; the PyTorch native update then runs
    # on the CPU reference device while the GEMS walk stays on the GPU.
    ref_scale = utils.to_reference(
        torch.tensor(scale_val, device=flag_gems.device, dtype=torch.float32)
    )
    ref_tracker = utils.to_reference(
        torch.tensor(0, device=flag_gems.device, dtype=torch.int32)
    )
    for _ in range(interval):
        step(False, ref_scale, ref_tracker)
    # After `interval` no-inf steps the scale should have doubled once and the
    # tracker reset to 0.
    expected_scale = scale_val * 2.0

    # GEMS walk
    res_scale = torch.tensor(scale_val, device=flag_gems.device, dtype=torch.float32)
    res_tracker = torch.tensor(0, device=flag_gems.device, dtype=torch.int32)
    for _ in range(interval):
        step(True, res_scale, res_tracker)

    utils.gems_assert_equal(res_scale, ref_scale)
    assert float(res_scale) == expected_scale
    assert int(res_tracker) == 0


@pytest.mark.amp_update_scale_
def test_amp_update_scale__backoff_then_recover():
    """Backoff on an inf, then grow back over an interval."""
    interval = 2

    # Reference.  ``to_reference`` moves the scalars to the CPU reference device
    # under ``--ref=cpu`` so the ``gems_assert_equal`` ``to_cpu`` contract holds;
    # the PyTorch native update then runs on the CPU reference device while the
    # GEMS walk stays on the GPU.
    ref_scale = utils.to_reference(
        torch.tensor(8.0, device=flag_gems.device, dtype=torch.float32)
    )
    ref_tracker = utils.to_reference(
        torch.tensor(0, device=flag_gems.device, dtype=torch.int32)
    )
    # Step 1: inf found -> backoff to 4.0, tracker reset.
    torch._amp_update_scale_(
        ref_scale,
        ref_tracker,
        utils.to_reference(
            torch.tensor(1.0, device=flag_gems.device, dtype=torch.float32)
        ),
        2.0,
        0.5,
        interval,
    )
    assert float(ref_scale) == 4.0
    # Step 2 & 3: no inf, tracker reaches interval -> grow to 8.0.
    for _ in range(interval):
        torch._amp_update_scale_(
            ref_scale,
            ref_tracker,
            utils.to_reference(
                torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)
            ),
            2.0,
            0.5,
            interval,
        )

    # GEMS mirrors the sequence
    res_scale = torch.tensor(8.0, device=flag_gems.device, dtype=torch.float32)
    res_tracker = torch.tensor(0, device=flag_gems.device, dtype=torch.int32)
    with flag_gems.use_gems():
        torch._amp_update_scale_(
            res_scale,
            res_tracker,
            torch.tensor(1.0, device=flag_gems.device, dtype=torch.float32),
            2.0,
            0.5,
            interval,
        )
        for _ in range(interval):
            torch._amp_update_scale_(
                res_scale,
                res_tracker,
                torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32),
                2.0,
                0.5,
                interval,
            )

    utils.gems_assert_equal(res_scale, ref_scale)
    utils.gems_assert_equal(res_tracker, ref_tracker)
    assert float(res_scale) == 8.0


@pytest.mark.amp_update_scale_
@pytest.mark.parametrize("interval", [1, 2, 5, 10])
def test_amp_update_scale__intervals(interval):
    """The scale grows exactly when the tracker is one below the interval."""
    # Tracker just below the interval -> grow.
    for tracker_val in (interval - 1, interval):
        # ``to_reference`` moves the scalars to the CPU reference device under
        # ``--ref=cpu`` so the ``gems_assert_equal`` ``to_cpu`` contract holds;
        # the PyTorch native update runs on the CPU reference device while the
        # GEMS update stays on the GPU.
        ref_scale = utils.to_reference(
            torch.tensor(100.0, device=flag_gems.device, dtype=torch.float32)
        )
        ref_tracker = utils.to_reference(
            torch.tensor(tracker_val, device=flag_gems.device, dtype=torch.int32)
        )
        ref_found_inf = utils.to_reference(
            torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)
        )
        torch._amp_update_scale_(
            ref_scale, ref_tracker, ref_found_inf, 3.0, 0.25, interval
        )

        res_scale = torch.tensor(100.0, device=flag_gems.device, dtype=torch.float32)
        res_tracker = torch.tensor(
            tracker_val, device=flag_gems.device, dtype=torch.int32
        )
        res_found_inf = torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)
        with flag_gems.use_gems():
            torch._amp_update_scale_(
                res_scale, res_tracker, res_found_inf, 3.0, 0.25, interval
            )

        utils.gems_assert_equal(res_scale, ref_scale)
        utils.gems_assert_equal(res_tracker, ref_tracker)
