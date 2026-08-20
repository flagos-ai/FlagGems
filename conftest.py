"""Strip editable install MetaPathFinder so tests load THIS worktree."""

import sys


def pytest_configure(config):
    # Remove editable-install path finders so this worktree's flag_gems loads
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not (
            hasattr(finder, "__class__")
            and (
                "EditableFinder" in finder.__class__.__name__
                or "ScikitBuildRedirectingFinder" in finder.__class__.__name__
            )
        )
    ]

    # Clear any already-imported flag_gems modules
    for key in list(sys.modules.keys()):
        if key == "flag_gems" or key.startswith("flag_gems."):
            del sys.modules[key]
