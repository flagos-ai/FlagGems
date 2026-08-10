"""Offline FlagTune collection, training, and reporting for FlagGems.

This package is intentionally separate from :mod:`flag_gems.utils`: it owns a
complete offline workflow rather than a reusable low-level helper.  Import the
library API from :mod:`flag_gems.flagtune.collection`; invoke user-facing
commands through the three ``flaggems-flagtune-*`` console scripts.
"""
