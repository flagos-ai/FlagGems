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

"""Plan B (custom ops): one module per operator, each carrying its native call
coordinates, shape grid, input construction and primary-key shape. The collector
(tools/collect_baseline_nvidia.py) is driven by the contract described below.

Each operator module must export:

    OP_NAME : str
        Operator name (written to JSON / report naming / matched against the
        operator name on the acceptance dashboard).
    DTYPES : list[torch.dtype]
        Data types to iterate over.
    IS_INPLACE : bool
        Whether native writes back into its inputs in place (in-place operators
        must rebuild their arguments before NCU sampling, to avoid warmup's
        in-place accumulation polluting the values).
    native() -> callable | None
        Resolve and return the NV native callable; return None if it cannot be
        resolved (the operator has no baseline and is skipped).
    grid() -> list[dict]
        List of collection-point bindings (the expanded dimension grid); each
        dict is one set of dimension values.
    build_inputs(binding, dtype, device) -> (args, kwargs)
        Construct the native call arguments following the corresponding
        benchmark's input_fn logic. native is called as op(*args, **kwargs).
    key_shape(binding) -> list[int] | str
        The primary key for this collection point in the JSON (corresponds to
        the benchmark's shape). Simple operators may return a shape list (e.g.
        [M, N]); complex operators (multiple tensors, irregular inputs) should
        return a semantic string that uniquely distinguishes the collection
        point (e.g. "bs4_sq2048_h32_hkv8_d128") to avoid key collisions across
        configurations.

Optional exports:

    config(binding, dtype) -> dict
        Used by complex operators to describe the real input/output shapes at
        this collection point, written to the JSON "config" field. Suggested
        structure: {"inputs": {...}, "outputs": {...}, "dims": {...}}, where
        each tensor entry looks like {"shape": [...], "dtype": "torch.bfloat16"}
        and each scalar entry looks like {"scalar": ...}. Simple operators
        (whose shape key is already expressive enough) may omit it.

Note: build_inputs is called once in the collector's main process and once in
the NCU subprocess (which re-imports this module by OP_NAME to rebuild inputs),
so it must be a side-effect-free, repeatably callable pure constructor.
"""
