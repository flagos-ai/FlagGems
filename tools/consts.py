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

DTYPE_MAP = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.int64": "int64",
    "torch.bool": "bool",
    "torch.complex64": "cf64",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
}
