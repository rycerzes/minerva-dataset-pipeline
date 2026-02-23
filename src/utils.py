from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from typing import TypeVar, Iterable


T = TypeVar("T", bound=BaseModel)


def model_to_parquet(models: list[T], output_path: str) -> None:
    records = [model.model_dump() for model in models]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)


def iter_to_parquet(models: Iterable[T], output_path: str) -> int:
    records = [model.model_dump() for model in models]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)
    return len(records)


def read_parquet(path: str, model_class: type[T]) -> list[T]:
    table = pq.read_table(path)
    return [model_class.model_validate(row) for row in table.to_pylist()]
