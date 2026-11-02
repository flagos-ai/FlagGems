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

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class BenchmarkCasePlan:
    """Tensor-free description produced by the case planning stage."""

    shape: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)
    builder_args: Tuple[Any, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    """Lightweight description of one benchmark case.

    ``builder_args`` are private runner state. They are intentionally excluded
    from the JSON case-list contract and are only consumed when the case is
    selected for execution.
    """

    case_id: str
    ordinal: int
    dtype: Any
    shape: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)
    builder_args: Tuple[Any, ...] = field(default_factory=tuple, repr=False)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "dtype": str(self.dtype),
            "shape": self.shape,
            "params": self.params,
        }


@dataclass(frozen=True)
class BenchmarkCaseList:
    op_name: str
    level: str
    cases: Tuple[BenchmarkCaseSpec, ...]
    phase: str = "timing"
    schema_version: str = "flaggems.benchmark-case-list/v2"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "op_name": self.op_name,
            "phase": self.phase,
            "level": self.level,
            "cases": [case.to_dict() for case in self.cases],
        }
