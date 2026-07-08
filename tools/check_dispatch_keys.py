#!/usr/bin/env python3
"""Validate that every dispatch key in ``_FULL_CONFIG`` names a real ATen overload.

FlagGems registers each kernel to PyTorch via ``torch.library.Library("aten",
"IMPL").impl(key, fn)``. ``torch.library`` silently accepts a key whose overload
does not exist in the ATen schema (e.g. ``greater.out`` instead of
``greater.Tensor_out``) but never dispatches to it, so the operator falls back to
eager and is not accelerated. This checker catches that whole class of typo.

Rules for a key ``base`` or ``base.overload``:
- ``base`` must be a real ``torch.ops.aten`` op, unless it is a FlagGems custom /
  fused op listed in ``NON_ATEN_OPS``.
- A bare key (no ``.overload``) registers at the operator level and is always
  valid once ``base`` exists.
- A dotted key must name an overload present in ``op.overloads()``, unless it is
  an intentional forward-compat key listed in ``FORWARD_COMPAT_KEYS`` (an
  overload that exists only on newer torch versions).

Usage:
    python3 tools/check_dispatch_keys.py            # report invalid keys
    python3 tools/check_dispatch_keys.py --check     # CI: exit 1 if any invalid
    python3 tools/check_dispatch_keys.py --list      # print every key + verdict
"""

from __future__ import annotations

import argparse
import sys

import torch

import flag_gems

# FlagGems custom / fused ops that are not part of torch.ops.aten. They have no
# ATen schema to validate against, so their keys are accepted as-is.
NON_ATEN_OPS = {
    "_grouped_mm",
    "_scaled_grouped_mm",
    "_scaled_grouped_mm",
    "add_rms_norm",
    "amin_",
    "beam_search_score",
    "beam_search_score_",
    "expand_",
    "grid_sample",
    "matmuladd",
    "max_pool3d_backward",
    "nll_loss_nd_backward",
    "nll_loss_nd_forward",
    "normed_cumsum",
    "scaled_softmax_backward",
    "scaled_softmax_forward",
    "special_chebyshev_polynomial_w_out",
    "special_i0e_out",
    "special_i1_out",
    "special_shifted_chebyshev_polynomial_u_",
}

# Dotted keys whose overload does not exist on the currently installed torch but
# is a deliberate forward-compat registration for newer torch versions.
FORWARD_COMPAT_KEYS = {
    "addmm.dtype",
    "addmm.dtype_out",
}


def iter_keys():
    """Yield every dispatch key string from _FULL_CONFIG."""
    for entry in flag_gems._FULL_CONFIG:
        yield entry[0]


def classify(key):
    """Return None if the key is valid, else a human-readable reason string."""
    base, _, overload = key.partition(".")

    if base in NON_ATEN_OPS:
        return None

    op = getattr(torch.ops.aten, base, None)
    if op is None:
        return (
            f"base op '{base}' is not in torch.ops.aten (add to NON_ATEN_OPS if custom)"
        )

    if not overload:
        # Bare key -> operator-level registration, valid once the op exists.
        return None

    if key in FORWARD_COMPAT_KEYS:
        return None

    overloads = op.overloads()
    if overload not in overloads:
        return f"overload '{overload}' does not exist; available: {overloads}"

    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any dispatch key is invalid (for CI)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every key with its verdict",
    )
    args = parser.parse_args(argv)

    invalid = []
    total = 0
    for key in iter_keys():
        total += 1
        reason = classify(key)
        if args.list:
            print(
                f"{'OK  ' if reason is None else 'BAD '}{key}"
                + (f"  <- {reason}" if reason else "")
            )
        if reason is not None:
            invalid.append((key, reason))

    if invalid:
        print(
            f"\n{len(invalid)} invalid dispatch key(s) out of {total}:", file=sys.stderr
        )
        for key, reason in invalid:
            print(f"  {key}: {reason}", file=sys.stderr)
        if args.check:
            return 1
    else:
        print(f"All {total} dispatch keys are valid.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
