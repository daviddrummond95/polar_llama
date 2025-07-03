from typing import Any
from polars import DataFrame, LazyFrame

def submit_df(plan: LazyFrame | DataFrame) -> DataFrame: ...