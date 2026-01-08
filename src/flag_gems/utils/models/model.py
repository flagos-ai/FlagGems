import inspect
from abc import abstractmethod
from typing import Dict, Final, Optional, Sequence, Tuple, Union, overload

import triton


class PersistantModel(object):
    signature: Final[inspect.Signature] = inspect.signature(triton.Config)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
    ) -> Optional[triton.Config]:
        ...

    @abstractmethod
    def get_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: triton.Config,
    ) -> Optional[Tuple[float, float, float]]:
        ...

    @overload
    def put_config(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: triton.Config,
    ) -> None:
        ...

    @overload
    def put_config(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Dict[str, Union[bool, int, float, str]],
    ) -> None:
        ...

    @abstractmethod
    def put_config(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Union[triton.Config, Dict[str, Union[bool, int, float, str]]],
    ) -> None:
        ...

    @overload
    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: triton.Config,
        benchmark: Tuple[float, float, float],
    ) -> None:
        ...

    @overload
    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Dict[str, Union[bool, int, float, str]],
        benchmark: Tuple[float, float, float],
    ) -> None:
        ...

    @abstractmethod
    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Union[triton.Config, Dict[str, Union[bool, int, float, str]]],
        benchmark: Tuple[float, float, float],
    ) -> None:
        ...
