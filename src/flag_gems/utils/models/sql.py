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

import os
from hashlib import md5
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import sqlalchemy
import sqlalchemy.ext.automap
import sqlalchemy.orm
import triton
from typing_extensions import override

from .model import PersistantModel
from .session import RollbackSession

# SQLAlchemy 2.0 introduced DeclarativeBase and mapped_column; LTS
# distributions (e.g. Ubuntu 22.04/24.04) still ship SQLAlchemy 1.4, so fall
# back to the legacy declarative API when they are unavailable.
_SQLALCHEMY_2 = hasattr(sqlalchemy.orm, "DeclarativeBase")

if _SQLALCHEMY_2:

    class Base(sqlalchemy.orm.DeclarativeBase): ...

else:
    Base = sqlalchemy.orm.declarative_base()


class SQLPersistantModel(PersistantModel):
    key_count_limit = 64 if "sqlite" in os.getenv("FLAGGEMS_DB_URL", "sqlite") else 32

    def __init__(self, db_url: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.engine: Final[sqlalchemy.engine.Engine] = sqlalchemy.create_engine(db_url)
        self.sql_model_pool: Dict[str, Type[Base]] = {}

    @staticmethod
    def _get_column_type(v: Union[Any, Type]) -> Any:
        """Determine the SQL column type for a given value or type.

        Uses BigInteger for integers to avoid overflow with large values
        (e.g. tensor element counts can exceed 2^31 - 1).
        """
        if isinstance(v, type):
            if issubclass(v, bool):
                return sqlalchemy.Boolean
            elif issubclass(v, int):
                return sqlalchemy.BigInteger
            elif issubclass(v, float):
                return sqlalchemy.Float
            elif issubclass(v, str):
                return sqlalchemy.String
        else:
            if isinstance(v, bool):
                return sqlalchemy.Boolean
            elif isinstance(v, int):
                return sqlalchemy.BigInteger
            elif isinstance(v, float):
                return sqlalchemy.Float
            elif isinstance(v, str):
                return sqlalchemy.String
        return None

    @staticmethod
    def build_sql_model_by_py(
        name: str,
        keys: Mapping[str, Union[Any, Type]],
        values: Mapping[str, Union[Any, Type]] = {},
    ) -> Type[Base]:
        key_count = len(keys)

        # Version-agnostic column specs: name -> (python type, SQL type, is_pk).
        # Rendered below either as SQLAlchemy 2.0 mapped_column() entries with
        # annotations, or as legacy 1.4 Column() declarations.
        specs: Dict[str, Tuple[type, Any, bool]] = {}

        if key_count > SQLPersistantModel.key_count_limit:
            # Blob mode: hash + blob
            specs["key_hash"] = (str, sqlalchemy.String(64), True)
            specs["key_blob"] = (str, sqlalchemy.Text, False)
            for k, v in chain(keys.items(), values.items()):
                if k in ("key_hash", "key_blob"):
                    continue
                specs[k] = (
                    v if isinstance(v, type) else type(v),
                    SQLPersistantModel._get_column_type(v),
                    k in keys.keys(),
                )
        else:
            # Column mode: individual key columns
            for k, v in chain(keys.items(), values.items()):
                specs[k] = (
                    v if isinstance(v, type) else type(v),
                    SQLPersistantModel._get_column_type(v),
                    k in keys.keys(),
                )

        if _SQLALCHEMY_2:
            annotations: Dict[str, type] = {
                k: sqlalchemy.orm.Mapped[py_type]
                for k, (py_type, _, _) in specs.items()
            }
            cols: Dict[str, sqlalchemy.orm.MappedColumn] = {
                k: sqlalchemy.orm.mapped_column(sql_type, primary_key=is_pk)
                for k, (_, sql_type, is_pk) in specs.items()
            }
            members: Dict[str, Any] = {"__annotations__": annotations, **cols}
        else:
            # SQLAlchemy 1.4 has no annotation-driven mapping; declare the
            # columns explicitly from the python types instead.
            members = {
                k: sqlalchemy.Column(sql_type, primary_key=is_pk)
                for k, (_, sql_type, is_pk) in specs.items()
            }

        ModelCls: Type[Base] = type(
            name,
            (Base,),
            {
                "__tablename__": name,
                "__table_args__": {"extend_existing": True},
                **members,
            },
        )
        return ModelCls

    @staticmethod
    def build_sql_model_by_db(
        name: str,
        engine: sqlalchemy.engine.Engine,
    ) -> Optional[Type[Base]]:
        AutoBase: sqlalchemy.ext.automap.AutomapBase = (
            sqlalchemy.ext.automap.automap_base()
        )
        AutoBase.prepare(engine)
        ModelCls: Optional[Type[Base]] = AutoBase.classes.get(name)
        return ModelCls

    @staticmethod
    def normalize_value(v: Any) -> Union[int, float, str]:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float, str)):
            return v
        return str(v)

    @staticmethod
    def get_key_dict(
        keys: Sequence[Union[bool, int, float, str]], use_blob_mode: bool = False
    ) -> Dict[str, Union[bool, int, float, str]]:
        if use_blob_mode:
            import hashlib

            # Create hash from all keys
            key_str = "_".join(str(k) for k in keys)
            key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:64]
            return {"key_hash": key_hash, "key_blob": key_str}
        else:
            return {
                f"key_{i}": SQLPersistantModel.normalize_value(v)
                for i, v in enumerate(keys)
            }

    @staticmethod
    def get_config_dict(
        config: triton.Config,
    ) -> Dict[str, Union[int, float, str]]:
        return {
            k: SQLPersistantModel.normalize_value(v)
            for k, v in config.all_kwargs().items()
            if isinstance(v, (int, float, str, bool))
        }

    def get_sql_model(
        self,
        name: str,
        keys: Mapping[str, Union[Any, Type]] = {},
        values: Mapping[str, Union[Any, Type]] = {},
    ) -> Callable[[str, Optional[Mapping[str, Type]]], Optional[Type[Base]]]:
        with self.lock:
            full_name = "{}-{}".format(
                name, md5("".join(keys.keys()).encode()).hexdigest()
            )
            db_url = str(self.engine.url)

            # Postgresql will chunk table name into 63-bytes string
            if "postgresql" in db_url:
                name_bytes = full_name.encode("ascii")
                if len(name_bytes) <= 63:
                    short_name = full_name
                else:
                    short_name = (
                        name_bytes[:30].decode() + "_" + md5(name_bytes).hexdigest()
                    )
            else:
                short_name = full_name

            ModelCls: Optional[Type[Base]] = self.sql_model_pool.get(short_name)
            if ModelCls is not None:
                return ModelCls
            ModelCls = SQLPersistantModel.build_sql_model_by_db(short_name, self.engine)
            if ModelCls is not None:
                self.sql_model_pool[short_name] = ModelCls
                return ModelCls
            if not keys or not values:
                return None
            ModelCls = SQLPersistantModel.build_sql_model_by_py(
                short_name, keys, values
            )
            try:
                with self.engine.begin() as conn:
                    conn.execute(
                        sqlalchemy.schema.CreateTable(
                            ModelCls.__table__, if_not_exists=True
                        )
                    )
            except Exception as e:
                # Because of concurrent execution, the same table/type name is
                # attempted to be created multiple times; these errors can be safely ignored.
                err_msg = str(e)
                if "duplicate key value violates unique constraint" not in err_msg:
                    # DuplicateObject (42710): e.g. type "xxx" already exists
                    if 'type "' not in err_msg or '" already exists' not in err_msg:
                        raise
            self.sql_model_pool[name] = ModelCls
            return ModelCls

    @override
    def get_config(
        self, name: str, keys: Sequence[Union[bool, int, float, str]]
    ) -> Optional[triton.Config]:
        # Detect if this table uses blob mode
        use_blob_mode = len(keys) > self.key_count_limit
        key_dict: Dict[str, Union[bool, int, float, str]] = (
            SQLPersistantModel.get_key_dict(keys, use_blob_mode)
        )
        ConfigCls: Optional[Type[Base]] = self.get_sql_model(name, key_dict)
        if ConfigCls is None:
            return None
        with RollbackSession(self.engine) as session:
            obj: Optional[Base] = session.get(
                ConfigCls,
                key_dict,
            )
            if obj is None:
                return None
            # Build obj_dict differently based on mode
            if use_blob_mode:
                # In blob mode, values are in obj properties
                obj_dict: Dict[str, Union[bool, int, float, str]] = {
                    k.key: getattr(obj, k.key)
                    for k in sqlalchemy.inspect(obj).mapper.columns
                    if k.key not in ["key_hash", "key_blob"]
                }
            else:
                obj_dict: Dict[str, Union[bool, int, float, str]] = {
                    k.key: getattr(obj, k.key)
                    for k in sqlalchemy.inspect(obj).mapper.columns
                    if k.key not in key_dict
                }
            kwargs: Dict[str, Union[bool, int, float, str]] = {
                k: v for k, v in obj_dict.items() if k not in self.signature.parameters
            }
            config_dict: Dict[str, int] = {
                k: v for k, v in obj_dict.items() if k in self.signature.parameters
            }
            return triton.Config(kwargs, **config_dict)

    @override
    def get_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: triton.Config,
    ) -> Optional[Tuple[float, float, float]]:
        config: Dict[str, Union[int, float, str]] = SQLPersistantModel.get_config_dict(
            config
        )

        use_blob_mode = len(keys) + len(config) > self.key_count_limit
        key_dict: Dict[str, Union[bool, int, float, str]] = {
            **SQLPersistantModel.get_key_dict(keys, use_blob_mode),
            **config,
        }
        BenchmarkCls: Optional[Type[Base]] = self.get_sql_model(name, key_dict)
        if BenchmarkCls is None:
            return None
        with RollbackSession(self.engine) as session:
            obj: Optional[Base] = session.get(
                BenchmarkCls,
                key_dict,
            )
            if obj is None:
                return None
            p50: float = obj.p50
            p20: float = obj.p20
            p80: float = obj.p80
            return (p50, p20, p80)

    def put_config(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Union[triton.Config, Dict[str, Union[bool, int, float, str]]],
    ) -> None:
        use_blob_mode = len(keys) > self.key_count_limit
        if isinstance(config, triton.Config):
            config: Dict[str, Union[int, float, str]] = (
                SQLPersistantModel.get_config_dict(config)
            )
        key_dict: Dict[str, Union[int, float, str]] = SQLPersistantModel.get_key_dict(
            keys, use_blob_mode
        )
        ConfigCls: Optional[Type[Base]] = self.get_sql_model(
            name,
            {k: type(v) for k, v in key_dict.items()},
            {k: type(v) for k, v in config.items()},
        )
        if ConfigCls is not None:
            with RollbackSession(self.engine) as session:
                existing: Optional[Base] = session.get(ConfigCls, key_dict)
                if existing is not None:
                    return
                session.add(ConfigCls(**key_dict, **config))
                session.commit()

    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Union[triton.Config, Dict[str, Union[bool, int, float, str]]],
        benchmark: Tuple[float, float, float],
    ) -> None:
        if isinstance(config, triton.Config):
            config: Dict[str, Union[int, float, str]] = (
                SQLPersistantModel.get_config_dict(config)
            )
        use_blob_mode = len(keys) + len(config) > self.key_count_limit
        key_dict: Dict[str, Union[int, float, str]] = SQLPersistantModel.get_key_dict(
            keys, use_blob_mode
        )
        p50, p20, p80 = benchmark
        benchmark: Dict[str, float] = {"p50": p50, "p20": p20, "p80": p80}
        BenchmarkCls: Optional[Type[Base]] = self.get_sql_model(
            name,
            key_dict | config,
            benchmark,
        )
        if BenchmarkCls is not None:
            with RollbackSession(self.engine) as session:
                benchmark_key_dict: Dict[str, Union[bool, int, float, str]] = (
                    key_dict | config
                )
                existing: Optional[Base] = session.get(BenchmarkCls, benchmark_key_dict)
                if existing is not None:
                    return
                session.add(BenchmarkCls(**benchmark_key_dict, **benchmark))
                session.commit()
