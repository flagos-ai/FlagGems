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


def backend_not_support(device_name, backend_list):
    raise RuntimeError(f"The {device_name} device is not supported currently. ")


def device_not_found():
    raise RuntimeError(
        "No device were detected on your machine ! \n "
        "Please check that your driver is complete. "
    )


def register_error(e):
    import logging

    logging.warning(f"Skipped registering operator: {e}")


def customized_op_replace_error(e):
    raise RuntimeError(
        e, "An exception occurred while replacing the customization operator."
    )
