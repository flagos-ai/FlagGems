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
