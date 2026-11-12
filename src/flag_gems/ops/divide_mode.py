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


# Resolve through the public namespace at call time: SpecOpRegistrar replaces
# these div entries with vendor/architecture implementations during package init.
# Importing directly from ops.div would bypass those specialized implementations.
def divide_scalar_mode(A, B, rounding_mode=None):
    import flag_gems

    return flag_gems.div_scalar_mode(A, B, rounding_mode=rounding_mode)


def divide__scalar_mode(A, B, rounding_mode=None):
    import flag_gems

    return flag_gems.div__scalar_mode(A, B, rounding_mode=rounding_mode)


def divide_tensor_mode(A, B, rounding_mode=None):
    import flag_gems

    return flag_gems.div_tensor_mode(A, B, rounding_mode=rounding_mode)


def divide__tensor_mode(A, B, rounding_mode=None):
    import flag_gems

    return flag_gems.div__tensor_mode(A, B, rounding_mode=rounding_mode)
