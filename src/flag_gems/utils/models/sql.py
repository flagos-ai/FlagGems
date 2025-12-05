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


class Base(sqlalchemy.orm.DeclarativeBase):
    ...


class SQLPersistantModel(PersistantModel):
    def __init__(self, db_url: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.engine: Final[sqlalchemy.engine.Engine] = sqlalchemy.create_engine(db_url)
        self.sql_model_pool: Dict[str, Type[Base]] = {}

    @staticmethod
    def build_sql_model_by_py(
        name: str,
        keys: Mapping[str, Union[Any, type]],
        values: Mapping[str, Union[Any, type]] = {},
    ) -> Type[Base]:
        annotations: Dict[str, type] = {
            k: sqlalchemy.orm.Mapped[v if isinstance(v, type) else type(v)]
            for k, v in chain(keys.items(), values.items())
        }
        cols: Dict[str, sqlalchemy.orm.MappedColumn] = {
            k: sqlalchemy.orm.mapped_column(primary_key=True) for k in keys.keys()
        } | {k: sqlalchemy.orm.mapped_column(primary_key=False) for k in values.keys()}
        ModelCls: Type[Base] = type(
            name,
            (Base,),
            {
                "__annotations__": annotations,
                "__tablename__": name,
                **cols,
            },
        )
        return ModelCls

    @staticmethod
    def build_sql_model_by_db(
        name: str,
        engine: sqlalchemy.engine.Engine,
    ) -> Optional[Type[Base]]:
        AutoBase: sqlalchemy.ext.automap.AutomapBase = (
            sqlalchemy.ext.automap.automap_base(Base)
        )
        AutoBase.prepare(engine)
        ModelCls: Optional[Type[Base]] = AutoBase.classes.get(name)
        return ModelCls

    @staticmethod
    def get_key_dict(
        keys: Sequence[Union[bool, int, float, str]],
    ) -> Dict[str, Union[bool, int, float, str]]:
        return {f"key_{i}": v for i, v in enumerate(keys)}

    @staticmethod
    def get_config_dict(
        config: triton.Config,
    ) -> Dict[str, Union[bool, int, float, str]]:
        return {
            k: v
            for k, v in config.all_kwargs().items()
            if isinstance(v, (int, float, str))
        }

    def get_sql_model(
        self,
        name: str,
        keys: Mapping[str, Union[Any, type]] = {},
        values: Mapping[str, Union[Any, type]] = {},
    ) -> Callable[[str, Optional[Mapping[str, type]]], Optional[Type[Base]]]:
        ModelCls: Optional[Type[Base]] = self.sql_model_pool.get(name)
        if ModelCls is not None:
            return ModelCls
        ModelCls: Optional[Type[Base]] = SQLPersistantModel.build_sql_model_by_db(
            name, self.engine
        )
        if ModelCls is not None:
            self.sql_model_pool[name] = ModelCls
            return ModelCls
        if not keys or not values:
            return None
        ModelCls: Type[Base] = SQLPersistantModel.build_sql_model_by_py(
            name, keys, values
        )
        with self.engine.begin() as conn:
            conn.execute(
                sqlalchemy.schema.CreateTable(ModelCls.__table__, if_not_exists=True)
            )
        self.sql_model_pool[name] = ModelCls
        return ModelCls

    @override
    def get_config(
        self, name: str, keys: Sequence[Union[bool, int, float, str]]
    ) -> Optional[triton.Config]:
        key_dict: Dict[
            str, Union[bool, int, float, str]
        ] = SQLPersistantModel.get_key_dict(keys)
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
        key_dict: Dict[str, Union[bool, int, float, str]] = {
            **SQLPersistantModel.get_key_dict(keys),
            **SQLPersistantModel.get_config_dict(config),
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
        if isinstance(config, triton.Config):
            config: Dict[
                str, Union[bool, int, float, str]
            ] = SQLPersistantModel.get_config_dict(config)
        key_dict: Dict[
            str, Union[bool, int, float, str]
        ] = SQLPersistantModel.get_key_dict(keys)
        ConfigCls: Optional[Type[Base]] = self.get_sql_model(
            name,
            {k: type(v) for k, v in key_dict.items()},
            {k: type(v) for k, v in config.items()},
        )
        if ConfigCls is not None:
            with RollbackSession(self.engine) as session:
                obj: Base = ConfigCls(**key_dict, **config)
                session.merge(obj)
                session.commit()

    def put_benchmark(
        self,
        name: str,
        keys: Sequence[Union[bool, int, float, str]],
        config: Union[triton.Config, Dict[str, Union[bool, int, float, str]]],
        benchmark: Tuple[float, float, float],
    ) -> None:
        key_dict: Dict[
            str, Union[bool, int, float, str]
        ] = SQLPersistantModel.get_key_dict(keys)
        if isinstance(config, triton.Config):
            config: Dict[
                str, Union[bool, int, float, str]
            ] = SQLPersistantModel.get_config_dict(config)
        p50, p20, p80 = benchmark
        benchmark: Dict[str, float] = {"p50": p50, "p20": p20, "p80": p80}
        BenchmarkCls: Optional[Type[Base]] = self.get_sql_model(
            name,
            key_dict | config,
            benchmark,
        )
        if BenchmarkCls is not None:
            with RollbackSession(self.engine) as session:
                obj: Base = BenchmarkCls(**key_dict, **config, **benchmark)
                session.merge(obj)
                session.commit()
