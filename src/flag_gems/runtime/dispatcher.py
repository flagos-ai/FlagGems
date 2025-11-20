import argparse
import os

import pytest

from . import backend
from .commom_utils import vendors


class OpDispatcher:
    """Operator Dispatcher for selecting vendor-specific operators and configurations."""

    def __init__(self, attrs=None):
        self.attrs = attrs or {}
        self.is_dispatch = False
        self.is_dispatch_configs = False
        self.is_dispatch_operators = False
        self.is_debug = False
        self.operator_vendor = None
        self.config_vendor = None
        self.configurations = None
        self.operators = None

        self._detect_user_action()
        if self.is_user_dispatch or attrs:
            self.is_dispatch = True
            self.set_init_attr()
            self._load_configs()
            self._load_device_info()

        if self.is_debug:
            self.debug_action()

    def debug_action(self):
        debug_prefix = "\033[93m[DEBUG]:"
        print(
            f" {debug_prefix} operator_vendor\033[0m : {self.operator_vendor}\n",
            f"{debug_prefix} operators\033[0m : {self.operators}\n",
            f"{debug_prefix} config_vendor\033[0m: {self.config_vendor}\n"
            f" {debug_prefix} configurations\033[0m : {self.configurations}\n"
            f" {debug_prefix} device_info\033[0m : {self.device_info}\n",
        )

    def _detect_user_action(self):
        """Detect whether user provided command-line args."""

        args = self.get_cmd_args()
        vendor = args.vendor
        self.is_debug = args.debug or self.is_debug
        if args.ops or args.configs or vendor:
            self.is_user_dispatch = True
            self.attrs = {
                "operators": vendor or args.ops,
                "configurations": vendor or args.configs,
            }
        else:
            self.is_user_dispatch = False

    def set_init_attr(self, attrs=None):
        attrs = attrs or self.attrs
        self.operator_vendor = attrs["operators"]
        self.config_vendor = attrs["configurations"]

    def _load_device_info(self):
        if self.operator_vendor is not None:
            all_info, failed_reasons = backend.get_vendor_infos(return_type="dict")
            if self.operator_vendor in failed_reasons:
                print(
                    f"\033[31m[Warning]\033[0m : \033[31mfailed to load vendor device info,"
                    f"the reason is {failed_reasons[self.operator_vendor]}\033[0m"
                )
                self.operator_vendor = None
                self.device_info = None
            else:
                self.device_info = all_info[self.operator_vendor]

    def _load_configs(self, config_vendor=None):
        """Load configuration sets for vendor."""

        vendor = self.config_vendor or config_vendor
        if vendor:
            self.configurations = {
                "heuristic": backend.get_heuristic_config(vendor_name=vendor),
                "autotune": backend.get_tune_config(vendor_name=vendor, query=True),
            }
            self.is_dispatch_configs = True
            print(
                f"\033[92m[INFO]\033[0m : \033[92m configurations of {vendor} vendor has been loaded\033[0m"
            )

    def _load_operators(self, operator_vendor=None):
        """Load operator implementations for vendor."""

        vendor = self.operator_vendor or operator_vendor
        if vendor:
            self.operators = backend.get_current_device_extend_op(vendor_name=vendor)
            self.is_dispatch_operators = True
            print(
                f"\033[92m[INFO]\033[0m :  \033[93mOperators of {vendor} vendor has been loaded\033[0m"
            )

    def get_cmd_args(self):
        parser = argparse.ArgumentParser(description="...")
        _vendors = list(vendors.get_all_vendors().keys())
        IS_PYTEST = "PYTEST_CURRENT_TEST" in os.environ
        parser.add_argument(
            "--ops",
            type=str,
            action="store",
            default=None,
            required=False,
            choices=_vendors,
            help="the operator provider(vendor) you want to specify",
        )
        parser.add_argument(
            "--configs",
            type=str,
            action="store",
            default=None,
            required=False,
            choices=_vendors,
            help="the configrations provider(vendor) you want to specify",
        )
        parser.add_argument(
            "--vendor",
            type=str,
            action="store",
            default=None,
            required=False,
            choices=_vendors,
            help="the configrations and the provider(vendor) you want to specify",
        )
        parser.add_argument(
            "--debug",
            type=str,
            action="store",
            default=None,
            required=False,
            choices=["False", "True"],
            help="debug for OpDispatcher",
        )
        args, remaining = parser.parse_known_args()
        if IS_PYTEST:
            pytest.main(remaining)
        return args


op_dispatcher = OpDispatcher()
