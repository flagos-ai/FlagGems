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

#pragma once

#include <cstddef>

namespace flag_gems::device {

struct DeviceInfo {
  int device_id;
  std::size_t l2_cache_size;
  int sm_count;
  int major;
};

const DeviceInfo &get_device_info(int device_id);
const DeviceInfo &get_current_device_info();

int current_device_id();
std::size_t current_l2_cache_size();
int current_sm_count();
int current_compute_capability_major();

}  // namespace flag_gems::device
