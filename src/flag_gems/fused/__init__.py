"""Compatibility shim: ``flag_gems.fused`` → ``flag_gems.deprecated``.

The kernel package was renamed from ``flag_gems.fused`` to
``flag_gems.deprecated``.  This shim aliases the old name to the new package
object so that all existing ``from flag_gems.fused import X`` /
``flag_gems.fused.X`` references keep working without modification.
New code should import from ``flag_gems.deprecated``.
"""

import sys

# Ensure the real package (and its submodule tree) is fully loaded first.
import flag_gems.deprecated  # noqa: F401

# Make ``flag_gems.fused`` literally be the ``flag_gems.deprecated`` module:
# every ``flag_gems.fused.X`` submodule import resolves to the same
# ``flag_gems.deprecated.X`` object, and ``flag_gems.fused.__path__`` points at
# the deprecated package, so submodules are found on disk without a finder.
sys.modules[__name__] = sys.modules["flag_gems.deprecated"]
