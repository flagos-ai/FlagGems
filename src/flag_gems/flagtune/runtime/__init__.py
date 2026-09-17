"""GPU-worker execution adapters for FlagGems FlagTune workflows.

Runtime code converts validated workload descriptions into tensors and calls
trusted public operators.  Scheduling, process ownership, and SQLite merging
are deliberately kept in :mod:`flag_gems.flagtune.collection`.
"""
