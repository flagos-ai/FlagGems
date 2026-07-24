#!/usr/bin/env python3
"""Validate that every dispatch key in ``_FULL_CONFIG`` names a real ATen overload.

FlagGems registers each kernel to PyTorch via ``torch.library.Library("aten",
"IMPL").impl(key, fn)``. ``torch.library`` silently accepts a key whose overload
does not exist in the ATen schema (e.g. ``greater.out`` instead of
``greater.Tensor_out``) but never dispatches to it, so the operator falls back to
eager and is not accelerated. This checker catches that whole class of typo.

Rules for a key ``base`` or ``base.overload``:
- If ``base`` is labeled ``fused`` in ``conf/operators.yaml`` it is a composite
  kernel, not an ATen op, so there is no overload to validate.
- Otherwise ``base`` must be a real ``torch.ops.aten`` op, unless it is one of
  the ``KNOWN_REGISTRY_DISCREPANCIES`` (registry entries that predate this check
  and still need cleanup).
- A bare key (no ``.overload``) registers at the operator level and is valid
  once ``base`` exists.
- A dotted key must name an overload present in ``op.overloads()``, unless it is
  an intentional forward-compat key in ``FORWARD_COMPAT_KEYS`` (an overload that
  exists only on newer torch versions).

Usage:
    python3 tools/check_dispatch_keys.py            # report invalid keys
    python3 tools/check_dispatch_keys.py --check     # CI: exit 1 if any invalid
    python3 tools/check_dispatch_keys.py --list      # print every key + verdict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

import flag_gems

ROOT = Path(__file__).resolve().parent.parent
OPERATORS_YAML = ROOT / "conf" / "operators.yaml"

# Registry discrepancies that are still tolerated by name. Each of these is
# registered under a base that is not in torch.ops.aten and is not labeled
# `fused` either: some are custom composites mislabeled `aten`, some use a
# non-standard name for a real ATen op (for instance `special_i0e_out` instead
# of the `special_i0e.out` overload). They should each be renamed to a valid
# overload, relabeled `fused`, or removed. This set is meant to shrink to empty
# over time; it is the concrete cleanup list for the operator registry.
KNOWN_REGISTRY_DISCREPANCIES = {
    "_grouped_mm",
    "_scaled_grouped_mm",
    "add_rms_norm",
    "amin_",
    "expand_",
    "grid_sample",
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


def load_fused_ops():
    """Return the set of operator ids labeled `fused` in operators.yaml."""
    with open(OPERATORS_YAML) as f:
        spec = yaml.safe_load(f)
    return {
        op["id"] for op in spec.get("ops", []) if "fused" in (op.get("labels") or [])
    }


def iter_keys():
    """Yield every dispatch key string from _FULL_CONFIG."""
    for entry in flag_gems._FULL_CONFIG:
        yield entry[0]


def classify(key, exempt_bases):
    """Return None if the key is valid, else a human-readable reason string."""
    base, _, overload = key.partition(".")

    if base in exempt_bases:
        return None

    op = getattr(torch.ops.aten, base, None)
    if op is None:
        return f"base op '{base}' is not in torch.ops.aten and is not labeled `fused`"

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

    exempt_bases = load_fused_ops() | KNOWN_REGISTRY_DISCREPANCIES

    invalid = []
    total = 0
    for key in iter_keys():
        total += 1
        reason = classify(key, exempt_bases)
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
