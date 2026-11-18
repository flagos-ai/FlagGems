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

"""Gems ``lift_fresh`` implementation.

``aten::lift_fresh(Tensor(a) self) -> Tensor(a)`` returns an alias of ``self``
with the exact same storage, layout, metadata and autograd state; no data is
materialized and write-through is bidirectional.  The reference-faithful
minimal form is therefore a single Python-level return of the input tensor
reference (same TensorImpl, same Python object identity, same leaf status).

Requirements (verified by the original Gems pytest):
  * out.untyped_storage().data_ptr() == self.untyped_storage().data_ptr()
  * storage_offset / sizes / strides / dtype / device / layout preserved
  * write-through in both directions (in-place updates visible in both)
  * requires_grad / is_leaf / grad_fn / version counter unchanged
  * bit-identical payloads for every dtype (NaN/Inf/signed-zero/fp8/bool/complex)

Any device kernel would have to materialize fresh storage, which contradicts
the alias/write-through contract and adds traffic, so this reviewed identity
operation is implemented without a JIT kernel (the bound Definition carries
``requires_triton_kernel: false``).
"""

__all__ = ["lift_fresh"]


def lift_fresh(self):
    return self
