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


def _broadcast_shape(tensors):
    max_ndim = max(tensor.ndim for tensor in tensors)
    shape = [1] * max_ndim

    for tensor in tensors:
        offset = max_ndim - tensor.ndim
        for index, size in enumerate(tensor.shape):
            dim = offset + index
            target = shape[dim]
            if size == target or size == 1:
                continue
            if target == 1:
                shape[dim] = size
                continue
            raise RuntimeError(
                "The size of tensor a ({}) must match the size of tensor b ({}) "
                "at non-singleton dimension {}".format(target, size, dim)
            )

    return tuple(shape)


def broadcast_tensors(*tensors):
    """Return broadcasted tensor views without materializing their contents."""
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if not tensors:
        return []
    if len(tensors) == 1:
        return [tensors[0]]

    target_shape = _broadcast_shape(tensors)
    results = []
    for tensor in tensors:
        leading_dims = len(target_shape) - tensor.ndim
        strides = [0] * leading_dims
        for size, target_size, stride in zip(
            tensor.shape, target_shape[leading_dims:], tensor.stride()
        ):
            strides.append(stride if size == target_size else 0)
        results.append(tensor.as_strided(target_shape, tuple(strides)))

    return results
