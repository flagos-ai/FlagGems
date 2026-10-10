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

# Public registration plus compatibility helpers used by MThreads baddbmm.
from .bmm import bmm as bmm
from .bmm import bmm_out as bmm_out
from .bmm import bmm_sqmma as bmm_sqmma  # noqa: F401
from .bmm import is_sqmma_compatible as is_sqmma_compatible  # noqa: F401

__all__ = ["bmm", "bmm_out"]
