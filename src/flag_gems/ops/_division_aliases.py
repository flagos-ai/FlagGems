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

"""Independent registration IDs for division overloads."""

from functools import wraps


def bind_division_aliases(registry, defaults=None):
    """Bind after specialization, preserving each overload's original target.

    Tensor true_divide historically used the tensor compatibility entry, not
    the shared true_divide entry. Snapshot targets before replacing names so
    those implementations (including backend overrides) remain independent.
    Explicit backend implementations of the canonical IDs take precedence.
    Legacy helpers retain their names and backend targets. Canonical IDs use
    key.lstrip('_').replace('.', '_').lower(), preserving internal underscores.
    """
    sources = {
        "div__scalar": "true_divide_",
        "div__scalar_mode": "div_mode_",
        "div__tensor": "true_divide_",
        "div__tensor_mode": "div_mode_",
        "div_out": "true_divide_out",
        "div_scalar": "true_divide",
        "div_scalar_mode": "div_mode",
        "div_tensor": "true_divide",
        "div_tensor_mode": "div_mode",
        "divide__scalar": "true_divide_",
        "divide__scalar_mode": "div_mode_",
        "divide__tensor": "true_divide_",
        "divide__tensor_mode": "div_mode_",
        "divide_scalar": "true_divide",
        "divide_scalar_mode": "div_mode",
        "divide_tensor": "divide",
        "divide_tensor_mode": "div_mode",
        "true_divide__scalar": "true_divide_",
        "true_divide__tensor": "true_divide_tensor_",
        "true_divide_scalar": "true_divide",
        "true_divide_tensor": "true_divide_tensor",
    }
    targets = {}
    for name, source in sources.items():
        if defaults is not None and registry.get(name) is not defaults.get(name):
            targets[name] = registry[name]
        else:
            targets[name] = registry[source]
    for name, target in targets.items():
        registry[name] = _entry(name, target)


def _entry(name, target):
    @wraps(target)
    def entry(*args, **kwargs):
        return target(*args, **kwargs)

    entry.__name__ = name
    entry.__qualname__ = name
    return entry
