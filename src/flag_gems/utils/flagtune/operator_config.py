"""Compile safe, data-driven operator planning and benchmark configuration.

The model-facing ``variants`` section is compiled by FlagTree.  The FlagGems
``pretune`` extension describes shape validation, ordered dispatch, tensor
construction, and public operator invocation without importing YAML-provided
Python objects or evaluating arbitrary source code.

Environment variables:
    ``FLAGGEMS_DB_URL`` is temporarily replaced with ``sqlite:///:memory:`` by
    :func:`initialize_planning_context` while importing and discovering the
    public operator. The original value is restored in ``finally``. This avoids
    accidental tuning-database writes during planning, but discovery may still
    initialize the active CUDA backend.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from triton.flagtune.expressions import (
    CompiledExpression,
    SafeExpressionError,
    SymbolRef,
    compile_expression,
    compile_reference,
    evaluate_compiled,
    require_mapping,
    validate_symbol_name,
)

from .records import PlanningContext, ShapeRecord


class OperatorConfigError(RuntimeError):
    """Report an invalid or unsupported operator benchmark configuration.

    The loader raises this user-facing exception for YAML syntax, schema,
    FlagTune model, public-operator, and planning-initialization failures.  It
    never embeds executable Python supplied by the YAML document.
    """


_MISSING = object()
_FIELD_KEYS = {"type", "required", "default", "min", "max", "aliases", "role"}
_TENSOR_KEYS = {"factory", "shape", "dtype"}
_FACTORIES = {"randn", "zeros", "ones", "empty"}
_PUBLIC_OPERATOR_OVERRIDES: dict[str, str] = {}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Require a mapping at one schema location and preserve its contents.

    Args:
        value: Decoded YAML value to validate.
        location: Human-readable path included in an error message.

    Returns:
        ``value`` narrowed to the mapping contract without copying it.

    Raises:
        OperatorConfigError: If the value is not a mapping.
    """
    try:
        return require_mapping(value, location)
    except SafeExpressionError as exc:
        raise OperatorConfigError(str(exc)) from exc


def _name(value: Any, location: str) -> str:
    """Validate a YAML name used as a field, tensor, or variant identifier.

    Args:
        value: Candidate decoded YAML scalar.
        location: Human-readable schema path used for diagnostics.

    Returns:
        The original non-empty Python identifier string.

    Raises:
        OperatorConfigError: If ``value`` is not an identifier.  Restricting
            names here also prevents attribute-path or expression syntax from
            entering later lookup stages.
    """
    try:
        return validate_symbol_name(value, location)
    except SafeExpressionError as exc:
        raise OperatorConfigError(str(exc)) from exc


@dataclass(frozen=True)
class ShapeFieldSpec:
    """Describe validation rules for one canonical workload field.

    Instances are compiled from ``pretune.shape.fields``.  ``default``,
    ``minimum``, and ``maximum`` may hold the private ``_MISSING`` sentinel so
    an explicit YAML null remains distinguishable from an omitted property.
    Only integer fields are currently supported.
    """

    name: str
    type_name: str
    required: bool
    default: Any
    minimum: Any
    maximum: Any
    aliases: tuple[str, ...]
    role: Optional[str]

    def normalize(self, value: Any, context: Mapping[str, Any], location: str) -> Any:
        """Convert and range-check one decoded shape value.

        Args:
            value: Scalar from a shape table, mapping case, or configured
                default.
            location: Fully qualified row/field path used in diagnostics.

        Returns:
            A normalized integer satisfying the configured inclusive bounds.

        Raises:
            OperatorConfigError: If the type is unsupported, conversion is
            lossy, a boolean is supplied as an integer, or a bound is violated.

        Notes:
            Integral floats and integer-like strings are accepted for input
            compatibility.  Non-integral floats are rejected explicitly.
        """
        if self.type_name == "int":
            if isinstance(value, bool):
                raise OperatorConfigError(f"{location} must be an integer, got boolean")
            try:
                converted = int(value)
            except (TypeError, ValueError) as exc:
                raise OperatorConfigError(
                    f"{location} must be an integer: {value!r}"
                ) from exc
            if isinstance(value, float) and not value.is_integer():
                raise OperatorConfigError(f"{location} must be an integer: {value!r}")
        else:
            raise OperatorConfigError(f"unsupported field type {self.type_name!r}")
        minimum = (
            evaluate_compiled(self.minimum, context, {})
            if self.minimum is not _MISSING
            else _MISSING
        )
        maximum = (
            evaluate_compiled(self.maximum, context, {})
            if self.maximum is not _MISSING
            else _MISSING
        )
        if minimum is not _MISSING and converted < minimum:
            raise OperatorConfigError(
                f"{location}={converted} is below minimum {minimum}"
            )
        if maximum is not _MISSING and converted > maximum:
            raise OperatorConfigError(
                f"{location}={converted} is above maximum {maximum}"
            )
        return converted


@dataclass(frozen=True)
class ShapeSchema:
    """Validate workload tables and preserve declared shape identity order.

    ``fields`` retains YAML declaration order, while ``identity`` defines the
    exact subset and order used for dispatch, tensor dimensions, shape keys, and
    output columns.  ``aliases`` provides case-insensitive input lookup only;
    normalized records always use canonical names.
    """

    fields: Mapping[str, ShapeFieldSpec]
    identity: tuple[str, ...]
    count_field: Optional[str]
    aliases: Mapping[str, str]

    def canonicalize_fields(self, raw_fields: Sequence[str]) -> tuple[str, ...]:
        """Resolve a shape-table header into canonical schema field names.

        Args:
            raw_fields: Ordered names from ``shape_spec`` or its accepted input
                alias.  Each name is stripped and matched case-insensitively.

        Returns:
            Canonical field names in the same order as the input header.

        Raises:
            OperatorConfigError: If a name is unknown or two header entries map
            to the same canonical field.
        """
        canonical = []
        for raw in raw_fields:
            key = str(raw).strip().lower()
            if key not in self.aliases:
                raise OperatorConfigError(f"unsupported shape field {raw!r}")
            canonical.append(self.aliases[key])
        if len(set(canonical)) != len(canonical):
            raise OperatorConfigError(
                f"shape specification contains duplicate fields: {canonical}"
            )
        return tuple(canonical)

    def normalize_values(
        self, raw_values: Mapping[str, Any], location: str
    ) -> tuple[dict[str, Any], Optional[int]]:
        """Normalize one canonical or aliased workload mapping.

        Args:
            raw_values: Field/value pairs from one shape row or API case.
            location: Row label used to produce actionable validation errors.

        Returns:
            ``(identity_values, count)``. ``identity_values`` contains exactly
            the declared identity fields in schema order; ``count`` is the
            optional normalized frequency field and is excluded from identity.

        Raises:
            OperatorConfigError: If an input is unknown or duplicated, a
            required field is absent, or field normalization fails.

        Notes:
            Non-identity auxiliary fields may be validated to support defaults
            and count extraction, but they are intentionally omitted from the
            returned identity mapping.
        """
        canonical: dict[str, Any] = {}
        for raw_name, value in raw_values.items():
            key = str(raw_name).strip().lower()
            if key not in self.aliases:
                raise OperatorConfigError(f"{location} has unknown field {raw_name!r}")
            name = self.aliases[key]
            if name in canonical:
                raise OperatorConfigError(f"{location} duplicates field {name!r}")
            canonical[name] = value

        normalized: dict[str, Any] = {}
        for name, field in self.fields.items():
            if name in canonical:
                value = canonical[name]
            elif field.default is not _MISSING:
                value = evaluate_compiled(field.default, normalized, {})
            elif field.required:
                raise OperatorConfigError(
                    f"{location} is missing required field {name!r}"
                )
            else:
                continue
            normalized[name] = field.normalize(value, normalized, f"{location}.{name}")

        identity = {name: normalized[name] for name in self.identity}
        count = normalized.get(self.count_field) if self.count_field else None
        return identity, count

    def build_records(self, shape_config: Any) -> list[ShapeRecord]:
        """Convert a parsed generic shape table into validated records.

        Args:
            shape_config: Object exposing ordered ``shape_spec`` and ``rows``
                attributes plus ``operator_name`` for diagnostics.  Pretune's
                generic YAML reader supplies this interface.

        Returns:
            One :class:`ShapeRecord` per input row, preserving source order.
            Variant and selected index remain unset for the selection stage.

        Raises:
            OperatorConfigError: If the header or any row violates this schema.

        Implementation:
            Header aliases are resolved once, each row is zipped to that header,
            and :meth:`normalize_values` projects the row into ordered identity
            values plus the optional workload count.
        """
        spec = self.canonicalize_fields(shape_config.shape_spec)
        records = []
        for index, row in enumerate(shape_config.rows):
            values, count = self.normalize_values(
                dict(zip(spec, row)), f"{shape_config.operator_name}.shapes[{index}]"
            )
            records.append(ShapeRecord(index, None, values, count))
        return records


@dataclass(frozen=True)
class TensorSpec:
    """Describe one safely constructible benchmark tensor.

    ``factory`` is restricted to a fixed torch-factory allowlist; every shape
    dimension is either a positive integer literal or a declared identity field.
    Device placement is deliberately absent from this configuration contract:
    the registered :class:`DeviceRuntime` owns it.  The executor accepts only
    ``dtype='runtime'`` so the caller's ordered dtype identity remains
    authoritative.
    """

    name: str
    factory: str
    shape: tuple[CompiledExpression, ...]
    dtype: str


@dataclass(frozen=True)
class BenchmarkSpec:
    """Describe tensor construction and a public FlagGems invocation.

    ``tensors`` defines trusted, allowlisted construction recipes. ``args`` is
    the ordered list of tensor names passed positionally to the public operator.
    Keyword arguments and YAML-provided callable paths are intentionally absent.
    """

    tensors: tuple[TensorSpec, ...]
    args: tuple[SymbolRef, ...]


@dataclass(frozen=True)
class OperatorBenchmarkSpec:
    """Hold compiled model, workload, dispatch, and execution contracts.

    The object is the single configuration artifact shared by planning and
    workers.  It combines FlagTree's parsed operator/variant metadata with the
    FlagGems-only shape and benchmark schema, plus a source hash that workers use
    to reject tasks prepared from a different revision of the YAML file.
    """

    source_path: Path
    source_sha256: str
    operator_info: Any
    shape: ShapeSchema
    dispatch_order: tuple[str, ...]
    benchmark: BenchmarkSpec

    @property
    def op_id(self) -> str:
        """Return the globally namespaced FlagTune operator identity."""
        return str(self.operator_info.op_id)

    @property
    def public_operator_name(self) -> str:
        """Derive the trusted public FlagGems callable name from ``op_id``.

        Returns:
            A Python identifier resolved from a code-owned override or the last
            path segment of a ``flaggems/...`` identity.

        Raises:
            OperatorConfigError: If the namespace is not handled by FlagGems.
        """
        return public_operator_name(self.op_id)

    def resolve_variant(self, values: Mapping[str, Any]) -> str:
        """Resolve an ordered shape mapping with first-match dispatch.

        Args:
            values: Canonical identity values accepted by variant predicates.

        Returns:
            The first variant whose compiled ``matches`` predicate succeeds,
            following ``pretune.dispatch.order`` exactly.

        Raises:
            OperatorConfigError: If no declared variant accepts the shape.

        Notes:
            Dispatch order is significant when predicates overlap.  The loader
            requires every model variant to appear exactly once in that order.
        """
        for name in self.dispatch_order:
            if self.operator_info.variants[name].matches(values):
                return name
        raise OperatorConfigError(
            f"no {self.op_id!r} variant matches shape {dict(values)!r}"
        )


def public_operator_name(op_id: str) -> str:
    """Resolve a FlagGems public callable name without YAML-provided imports."""
    try:
        from triton.flagtune.registry import validate_op_id

        validated = validate_op_id(op_id)
    except (ImportError, TypeError, ValueError) as exc:
        raise OperatorConfigError(f"invalid FlagTune op_id: {exc}") from exc
    parts = validated.split("/")
    if parts[0] != "flaggems":
        raise OperatorConfigError(
            f"unsupported FlagTune op_id namespace {parts[0]!r}; expected 'flaggems'"
        )
    name = _PUBLIC_OPERATOR_OVERRIDES.get(validated, parts[-1])
    return _name(name, f"public operator for {validated!r}")


def resolve_public_operator(flag_gems_module: Any, op_id: str) -> Any:
    """Return the trusted public FlagGems callable derived from ``op_id``."""
    name = public_operator_name(op_id)
    operator = getattr(flag_gems_module, name, None)
    if not callable(operator):
        raise OperatorConfigError(
            f"flag_gems has no public callable {name!r} for op_id {op_id!r}"
        )
    return operator


def _parse_shape(raw: Any, location: str) -> ShapeSchema:
    """Compile the safe ``pretune.shape`` YAML section.

    Args:
        raw: Decoded shape-section value.
        location: Root schema path used in all diagnostics.

    Returns:
        A :class:`ShapeSchema` with validated fields, aliases, identity order,
        defaults, bounds, and at most one optional count field.

    Raises:
        OperatorConfigError: If keys, names, field definitions, aliases, roles,
        or identity declarations violate the version-2 schema.

    Implementation:
        The parser rejects unknown keys before constructing immutable field
        records.  Alias uniqueness is checked case-insensitively, and identity
        fields must be declared, unique, and always available.
    """
    root = _mapping(raw, location)
    unknown = set(root) - {"fields", "identity"}
    if unknown:
        raise OperatorConfigError(f"{location} has unknown keys: {sorted(unknown)}")
    raw_fields = _mapping(root.get("fields"), f"{location}.fields")
    if not raw_fields:
        raise OperatorConfigError(f"{location}.fields must not be empty")
    fields: dict[str, ShapeFieldSpec] = {}
    aliases: dict[str, str] = {}
    count_field = None
    available_fields: set[str] = set()
    for raw_name, raw_field in raw_fields.items():
        name = _name(raw_name, f"{location}.fields key")
        field = _mapping(raw_field, f"{location}.fields.{name}")
        unknown = set(field) - _FIELD_KEYS
        if unknown:
            raise OperatorConfigError(
                f"{location}.fields.{name} has unknown keys: {sorted(unknown)}"
            )
        type_name = str(field.get("type", "int"))
        required = bool(field.get("required", "default" not in field))
        try:
            default = (
                compile_expression(
                    field["default"],
                    symbols=available_fields,
                    operations={},
                    location=f"{location}.fields.{name}.default",
                    allow_calls=False,
                )
                if "default" in field
                else _MISSING
            )
            minimum = (
                compile_expression(
                    field["min"],
                    symbols=available_fields,
                    operations={},
                    location=f"{location}.fields.{name}.min",
                    allow_calls=False,
                )
                if "min" in field
                else _MISSING
            )
            maximum = (
                compile_expression(
                    field["max"],
                    symbols=available_fields,
                    operations={},
                    location=f"{location}.fields.{name}.max",
                    allow_calls=False,
                )
                if "max" in field
                else _MISSING
            )
        except SafeExpressionError as exc:
            raise OperatorConfigError(str(exc)) from exc
        raw_aliases = field.get("aliases", [])
        if not isinstance(raw_aliases, list) or not all(
            isinstance(item, str) and item for item in raw_aliases
        ):
            raise OperatorConfigError(
                f"{location}.fields.{name}.aliases must be a list of names"
            )
        role = field.get("role")
        if role not in (None, "count"):
            raise OperatorConfigError(f"{location}.fields.{name}.role must be 'count'")
        if role == "count":
            if count_field is not None:
                raise OperatorConfigError(f"{location} defines multiple count fields")
            count_field = name
        spec = ShapeFieldSpec(
            name,
            type_name,
            required,
            default,
            minimum,
            maximum,
            tuple(raw_aliases),
            role,
        )
        for alias in (name, *raw_aliases):
            key = alias.lower()
            if key in aliases:
                raise OperatorConfigError(f"{location} duplicates alias {alias!r}")
            aliases[key] = name
        fields[name] = spec
        available_fields.add(name)

    raw_identity = root.get("identity")
    if not isinstance(raw_identity, list) or not raw_identity:
        raise OperatorConfigError(f"{location}.identity must be a non-empty list")
    identity = tuple(_name(item, f"{location}.identity") for item in raw_identity)
    if len(set(identity)) != len(identity) or any(
        name not in fields for name in identity
    ):
        raise OperatorConfigError(
            f"{location}.identity must contain unique declared fields"
        )
    if count_field in identity:
        raise OperatorConfigError(
            f"{location}.identity must not contain the count field"
        )
    for name in identity:
        field = fields[name]
        if not field.required and field.default is _MISSING:
            raise OperatorConfigError(
                f"identity field {name!r} must be required or defaulted"
            )
    return ShapeSchema(fields, identity, count_field, aliases)


def _parse_benchmark(raw: Any, shape: ShapeSchema, location: str) -> BenchmarkSpec:
    """Compile allowlisted tensor recipes and a public invocation contract.

    Args:
        raw: Decoded ``pretune.benchmark`` YAML section.
        shape: Previously compiled schema used to validate symbolic dimensions.
        location: Root schema path used in diagnostics.

    Returns:
        A :class:`BenchmarkSpec` ready for the generic GPU executor.

    Raises:
        OperatorConfigError: If unknown keys, unsupported factories, invalid
        dimensions, non-runtime dtype values, or undeclared invocation
        arguments are present.

    Security:
        Tensor factories come from ``_FACTORIES`` and invocation kind must be
        ``flag_gems_public``.  YAML cannot name Python modules, arbitrary
        callables, keyword expressions, or code to evaluate.
    """
    root = _mapping(raw, location)
    unknown = set(root) - {"tensors", "invoke"}
    if unknown:
        raise OperatorConfigError(f"{location} has unknown keys: {sorted(unknown)}")
    raw_tensors = _mapping(root.get("tensors"), f"{location}.tensors")
    tensors = []
    for raw_name, raw_tensor in raw_tensors.items():
        name = _name(raw_name, f"{location}.tensors key")
        tensor = _mapping(raw_tensor, f"{location}.tensors.{name}")
        unknown = set(tensor) - _TENSOR_KEYS
        if unknown:
            raise OperatorConfigError(
                f"{location}.tensors.{name} has unknown keys: {sorted(unknown)}"
            )
        factory = str(tensor.get("factory"))
        if factory not in _FACTORIES:
            raise OperatorConfigError(
                f"{location}.tensors.{name}.factory must be one of {sorted(_FACTORIES)}"
            )
        raw_dims = tensor.get("shape")
        if not isinstance(raw_dims, list) or not raw_dims:
            raise OperatorConfigError(f"{location}.tensors.{name}.shape must be a list")
        dims = []
        for index, dim in enumerate(raw_dims):
            if isinstance(dim, bool):
                raise OperatorConfigError(
                    f"{location}.tensors.{name}.shape dimensions must not be boolean"
                )
            if isinstance(dim, int) and dim <= 0:
                raise OperatorConfigError(
                    f"{location}.tensors.{name}.shape dimensions must be positive"
                )
            try:
                dims.append(
                    compile_expression(
                        dim,
                        symbols=set(shape.identity),
                        operations={},
                        location=f"{location}.tensors.{name}.shape[{index}]",
                        allow_calls=False,
                    )
                )
            except SafeExpressionError as exc:
                raise OperatorConfigError(str(exc)) from exc
        dtype = str(tensor.get("dtype", "runtime"))
        if dtype != "runtime":
            raise OperatorConfigError(
                f"{location}.tensors.{name} supports only dtype=runtime"
            )
        tensors.append(TensorSpec(name, factory, tuple(dims), dtype))
    if not tensors:
        raise OperatorConfigError(f"{location}.tensors must not be empty")

    invoke = _mapping(root.get("invoke"), f"{location}.invoke")
    unknown = set(invoke) - {"kind", "args"}
    if unknown:
        raise OperatorConfigError(
            f"{location}.invoke has unknown keys: {sorted(unknown)}"
        )
    if invoke.get("kind") != "flag_gems_public":
        raise OperatorConfigError(f"{location}.invoke.kind must be 'flag_gems_public'")
    args = invoke.get("args")
    tensor_names = {tensor.name for tensor in tensors}
    if not isinstance(args, list) or not args:
        raise OperatorConfigError(
            f"{location}.invoke.args must reference declared tensors"
        )
    try:
        compiled_args = tuple(
            compile_reference(
                item,
                symbols=tensor_names,
                location=f"{location}.invoke.args[{index}]",
            )
            for index, item in enumerate(args)
        )
    except SafeExpressionError as exc:
        raise OperatorConfigError(str(exc)) from exc
    return BenchmarkSpec(tuple(tensors), compiled_args)


def load_operator_benchmark_spec(path: str | Path) -> OperatorBenchmarkSpec:
    """Safely load and compile one version-3 FlagTune operator configuration.

    Args:
        path: Filesystem path to the combined FlagTree model and FlagGems
            ``pretune`` YAML document. User-home expansion is supported.

    Returns:
        An immutable :class:`OperatorBenchmarkSpec` containing compiled variant,
        shape, dispatch, benchmark, source-path, and SHA-256 metadata.

    Raises:
        OperatorConfigError: If PyYAML is unavailable, the file is missing,
        YAML decoding fails, schema version or keys are invalid, FlagTree rejects
        the variants, or any Pretune subsection violates its safe schema.

    Implementation:
        The document is read as bytes, decoded only with ``yaml.safe_load``, and
        checked with strict unknown-key rejection.  FlagTree compiles only the
        model-facing ``op_id`` and ``variants`` sections; local parsers compile
        shape and benchmark behavior.  A hash of the original bytes is retained
        to detect planner/worker configuration drift.

    Limitations:
        Only schema version 3, ``first_match`` dispatch, integer shape fields,
        allowlisted torch factories, CUDA runtime dtype, and positional public
        FlagGems calls are currently supported.
    """
    try:
        import yaml
    except ImportError as exc:
        raise OperatorConfigError("FlagTune YAML loading requires PyYAML") from exc
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise OperatorConfigError(f"operator config does not exist: {config_path}")
    source = config_path.read_bytes()
    try:
        root = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise OperatorConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(root, "config")
    unknown = set(root) - {"schema_version", "op_id", "variants", "pretune"}
    if unknown:
        raise OperatorConfigError(f"config has unknown keys: {sorted(unknown)}")
    if root.get("schema_version") != 3:
        raise OperatorConfigError("config.schema_version must be 3")
    try:
        from triton.flagtune.registry import parse_operator_config

        operator_info = parse_operator_config(
            {"op_id": root.get("op_id"), "variants": root.get("variants")}
        )
        public_operator_name(operator_info.op_id)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise OperatorConfigError(f"invalid FlagTune variants: {exc}") from exc
    pretune = _mapping(root.get("pretune"), "config.pretune")
    unknown = set(pretune) - {"shape", "dispatch", "benchmark"}
    if unknown:
        raise OperatorConfigError(f"config.pretune has unknown keys: {sorted(unknown)}")
    shape = _parse_shape(pretune.get("shape"), "config.pretune.shape")
    dispatch = _mapping(pretune.get("dispatch"), "config.pretune.dispatch")
    unknown = set(dispatch) - {"policy", "order"}
    if unknown:
        raise OperatorConfigError(
            f"config.pretune.dispatch has unknown keys: {sorted(unknown)}"
        )
    if dispatch.get("policy") != "first_match":
        raise OperatorConfigError(
            "config.pretune.dispatch.policy must be 'first_match'"
        )
    raw_order = dispatch.get("order")
    if not isinstance(raw_order, list):
        raise OperatorConfigError("config.pretune.dispatch.order must be a list")
    order = tuple(_name(item, "config.pretune.dispatch.order") for item in raw_order)
    if len(set(order)) != len(order) or set(order) != set(operator_info.variants):
        raise OperatorConfigError(
            "config.pretune.dispatch.order must list every variant exactly once"
        )
    benchmark = _parse_benchmark(
        pretune.get("benchmark"), shape, "config.pretune.benchmark"
    )
    return OperatorBenchmarkSpec(
        config_path,
        hashlib.sha256(source).hexdigest(),
        operator_info,
        shape,
        order,
        benchmark,
    )


def initialize_planning_context(
    spec: OperatorBenchmarkSpec,
) -> tuple[PlanningContext, Any]:
    """Initialize backend metadata for a configured public FlagGems operator.

    Args:
        spec: Fully compiled operator benchmark contract.

    Returns:
        ``(planning_context, operator_info)``.  The context records active
        vendor, Triton major/minor version, process-visible devices, and
        configured variants; ``operator_info`` is the FlagTree object already
        held by ``spec`` for downstream variant/config generation.

    Raises:
        OperatorConfigError: If dependencies cannot import, the public operator
        is absent, or registered-backend metadata cannot be queried.

    Implementation:
        Initialization temporarily points ``FLAGGEMS_DB_URL`` at an in-memory
        SQLite database so discovery cannot create or modify the caller's normal
        tuning database.  The original environment value is restored in a
        ``finally`` block even when initialization fails.

    Limitations:
        This call imports FlagGems and initializes the registered device
        backend. It discovers visibility but does not inspect occupancy or
        reserve a device.
    """
    original_db_url = os.environ.get("FLAGGEMS_DB_URL")
    os.environ["FLAGGEMS_DB_URL"] = "sqlite:///:memory:"
    try:
        import triton

        import flag_gems
        from flag_gems.utils.flagtune.device_runtime import (
            probe_flagtune_environment,
        )
        from flag_gems.runtime.backend import _state

        resolve_public_operator(flag_gems, spec.op_id)
        version = triton.__version__.split(".")
        environment = probe_flagtune_environment()
        device_tokens = environment.runtime.visible_device_tokens(
            environment.device_count
        )
        context = PlanningContext(
            operator_name=spec.public_operator_name,
            operator_variants=tuple(spec.operator_info.variants),
            vendor_name=str(_state.vendor_module.vendor_info.vendor_name),
            backend_name=environment.runtime.backend,
            triton_major=int(version[0]),
            triton_minor=int(version[1]),
            visible_device_count=environment.device_count,
            device_names=tuple(item.device_name for item in environment.devices),
            device_architectures=tuple(
                item.architecture for item in environment.devices
            ),
            device_tokens=tuple(device_tokens),
        )
        return context, spec.operator_info
    except Exception as exc:
        if isinstance(exc, OperatorConfigError):
            raise
        raise OperatorConfigError(
            f"cannot initialize FlagGems planning metadata: {exc}"
        ) from exc
    finally:
        if original_db_url is None:
            os.environ.pop("FLAGGEMS_DB_URL", None)
        else:
            os.environ["FLAGGEMS_DB_URL"] = original_db_url
