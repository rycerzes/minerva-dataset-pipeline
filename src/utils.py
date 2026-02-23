from __future__ import annotations

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from typing import TypeVar, Iterable, Optional, Any


T = TypeVar("T", bound=BaseModel)


def download(url: str, timeout: float = 60.0) -> httpx.Response:
    client = httpx.Client(timeout=timeout)
    try:
        response = client.get(url)
        response.raise_for_status()
        return response
    finally:
        client.close()


def save(content: Any, output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write(str(content))


def to_parquet(records: list[dict], output_path: str) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)


def model_to_parquet(models: list[T], output_path: str) -> None:
    records = [model.model_dump() for model in models]
    to_parquet(records, output_path)


def iter_to_parquet(models: Iterable[T], output_path: str) -> int:
    records = [model.model_dump() for model in models]
    to_parquet(records, output_path)
    return len(records)


def read_parquet(path: str, model_class: type[T]) -> list[T]:
    table = pq.read_table(path)
    return [model_class.model_validate(row) for row in table.to_pylist()]
