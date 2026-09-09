"""Define operator-independent records shared by FlagTune planning stages.

The records in this module are deliberately free of tensor, kernel, and
operator-specific behavior.  The CLI planner creates them from validated shape
tables, selection attaches a variant and stable index, and the benchmark
scheduler serializes them into worker task files.

All shape identity ordering comes from the insertion order of ``values``.  A
caller must therefore construct records through the compiled operator schema or
otherwise preserve the schema's declared identity order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ShapeRecord:
    """Represent one workload shape throughout selection and benchmarking.

    Args:
        source_index: Zero-based row number in the input shape table.
        selected_index: Zero-based position after filtering and sorting, or
            ``None`` before selection has assigned a position.
        values: Canonical identity-field mapping in the exact order declared by
            the operator YAML schema.
        count: Optional workload frequency used for reporting; it is not part
            of shape identity or variant dispatch.
        variant: Resolved FlagTune variant name, or ``None`` before dispatch.

    Notes:
        The dataclass is frozen to prevent stages from silently changing a
        record in place.  The nested ``values`` dictionary remains mutable by
        Python convention, so callers should treat it as immutable as well.
    """

    source_index: int
    selected_index: Optional[int]
    values: dict[str, Any]
    count: Optional[int]
    variant: Optional[str] = None

    @property
    def shape(self) -> list[Any]:
        """Return the ordered shape identity as a new list.

        Returns:
            Values from :attr:`values` in mapping insertion order.  The result
            can be changed without mutating the record.

        Notes:
            This property does not sort field names.  Schema construction is
            responsible for preserving the YAML ``identity`` order.
        """
        return list(self.values.values())

    @property
    def shape_key(self) -> str:
        """Return a compact comma-separated identity for logs and result rows.

        Returns:
            The string form of every ordered shape value joined by commas.

        Notes:
            This is a human-readable workload key, not a reversible serializer;
            code that needs field names or typed values should use
            :attr:`values` or :meth:`to_json`.
        """
        return ",".join(str(value) for value in self.shape)

    def to_json(self) -> dict[str, Any]:
        """Convert the record into the mapping used by worker task JSON.

        Returns:
            A newly allocated dictionary containing every dataclass field.

        Notes:
            Schema-normalized records currently contain JSON scalar values.
            This method uses :func:`dataclasses.asdict` but does not provide a
            general encoder for arbitrary objects inserted by external callers.
        """
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ShapeRecord":
        """Reconstruct a record from a decoded :meth:`to_json` payload.

        Args:
            payload: Mapping containing the five ``ShapeRecord`` fields.  It is
                normally produced by decoding a scheduler task file.

        Returns:
            A new frozen record with a private copy of the nested ``values``
            mapping.

        Raises:
            KeyError: If ``values`` is absent.
            TypeError: If fields are missing, unexpected, or incompatible with
                the dataclass constructor.

        Notes:
            Validation and type normalization belong to ``ShapeSchema``.  This
            transport helper intentionally trusts an already prepared payload.
        """
        values = dict(payload)
        values["values"] = dict(values["values"])
        return cls(**values)

    def to_benchmark_shape(self) -> dict[str, Any]:
        """Return the opaque shape mapping accepted by the batch scheduler.

        Returns:
            The same JSON-compatible representation as :meth:`to_json`.

        Usage:
            Pretune passes this mapping as ``BenchmarkCase.shape``.  The generic
            executor later revalidates its values against the operator schema
            before a GPU worker runs.
        """
        return self.to_json()


@dataclass(frozen=True)
class PlanningContext:
    """Describe runtime facts needed before scheduling GPU workers.

    Args:
        operator_name: Public callable name exported by ``flag_gems``.
        operator_variants: Variant names compiled from the operator YAML.
        vendor_name: Active FlagGems backend vendor identifier.
        triton_major: Major component of the imported Triton version.
        triton_minor: Minor component of the imported Triton version.
        backend_name: Active Triton backend identifier.
        visible_device_count: Device count visible to the planning process.
        device_names: Product names in logical-device order.
        device_architectures: Backend-native targets in logical-device order.
        device_tokens: Launcher tokens mapped from the active visibility policy.

    Notes:
        This object contains discovery metadata only.  It neither initializes
        benchmark workers nor guarantees that every visible device is idle or
        usable; GPU selection and occupancy checks remain caller concerns.
    """

    operator_name: str
    operator_variants: tuple[str, ...]
    vendor_name: str
    backend_name: str
    triton_major: int
    triton_minor: int
    visible_device_count: int
    device_names: tuple[str, ...]
    device_architectures: tuple[str, ...]
    device_tokens: tuple[str, ...]
