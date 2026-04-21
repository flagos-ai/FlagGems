#pragma once
#include <string>
#include <vector>

extern std::vector<std::string> registered_ops;

void register_cpp_ops(const std::vector<std::string>& op_names);
void register_all_cpp_ops();

std::vector<std::string> get_available_ops();
std::vector<std::string> get_registered_ops();
