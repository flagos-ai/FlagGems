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

import inspect
import threading
from abc import abstractmethod
from typing import Dict, Final, Optional, Sequence, Tuple, Union, overload

import triton


class PersistantModel(object):
    signature: Final[inspect.Signature] = inspect.signature(triton.Config)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lock: Final[threading.Lock] = threading.Lock()

    @staticmethod
    def parse_config(config: triton.Config) -> Dict[str, Union[int, float, str]]:
        return {
            k: v
            for k, v in config.all_kwargs().items()
            if isinstance(v, (int, float, str))
        }

    @abstractmethod
    def get_config(
        self, name: str, key: Sequence[Union[bool, int, float, str]]
    ) -> Optional[triton.Config]: ...

    @abstractmethod
    def get_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: triton.Config,
    ) -> Optional[Tuple[float, float, float]]: ...

    @overload
    def put_config(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: triton.Config,
    ) -> None: ...

    @overload
    def put_config(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Dict[str, Union[bool, int, float, str]],
    ) -> None: ...

    @abstractmethod
    def put_config(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Union[triton.Config, Dict[str, Union[bool, int, float, str]]],
    ) -> None: ...

    @overload
    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: triton.Config,
        benchmark: Tuple[float, float, float],
    ) -> None: ...

    @overload
    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Dict[str, Union[bool, int, float, str]],
        benchmark: Tuple[float, float, float],
    ) -> None: ...

    @abstractmethod
    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Union[triton.Config, Dict[str, Union[bool, int, float, str]]],
        benchmark: Tuple[float, float, float],
    ) -> None: ...
