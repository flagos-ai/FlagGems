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

"""Public ``topk_w8a16_fp8`` entry.

Hygon (E4M3FN) and THead / PPU (E5M2) implementations live in their
backend ops packages and are installed over this stub by ``SpecOpRegistrar``.
Other vendors should add their own backend implementation.
"""


def topk_w8a16_fp8(*args, **kwargs):
    raise NotImplementedError(
        "topk_w8a16_fp8 is implemented for the Hygon and THead/PPU backends; "
        "import flag_gems.topk_w8a16_fp8 after the vendor registrar has run"
    )
