# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest extension point for candidate-only profiling capture."""

from __future__ import annotations

import pytest


class ProfileHooks:
    @pytest.hookspec(firstresult=True)
    def pytest_flaggems_profile_scope(self, backend, case_id):
        """Return a context manager around capture, or None for plain replay.

        Called after input preparation, warmup and device synchronization.
        The context covers candidate iterations and their final synchronization.
        Implementations must release capture resources when the candidate fails.
        """


__all__ = ["ProfileHooks"]
