// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Definition unit for the generic boxed pointwise adapter. This is the ONLY
// translation unit that pulls in pointwise_prepare_args.h (and its transitive
// CUDA / triton_jit headers), so it must be compiled as part of the
// `operators` library, which owns those include paths. cstub.cpp only sees the
// declaration in pointwise_boxed.h.
#define FLAGGEMS_POINTWISE_BOXED_IMPL
#include "pointwise_boxed.h"
