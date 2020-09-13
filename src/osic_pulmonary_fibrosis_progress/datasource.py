from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class DataSource:
    df: pd.DataFrame
    roots: Union[Path, Dict[str, Path]]
    folds: List[int]

    def __len__(self) -> int:
        return len(self.df)


def train_validate_split(
    source: DataSource, val_folds: List[int] = [0, 1]
) -> Tuple[DataSource, DataSource]:
    train_folds = get_complimental_folds(val_folds)
    train_df = cast(pd.DataFrame, source.df[source.df["fold"].isin(train_folds)])
    val_df = cast(pd.DataFrame, source.df[source.df["fold"].isin(val_folds)])
    return (
        DataSource(train_df, source.roots, train_folds),
        DataSource(val_df, source.roots, val_folds),
    )


def kfold_split(source: DataSource, n_fold=4):
    if n_fold not in [4, 8]:
        raise ValueError("n_fold must be 4 or 8")

    for train_folds, val_folds in KFold(n_fold).split(range(8)):
        train_df = cast(pd.DataFrame, source.df[source.df["fold"].isin(train_folds)])
        val_df = cast(pd.DataFrame, source.df[source.df["fold"].isin(val_folds)])
        train_source = DataSource(train_df, source.roots, train_folds)
        val_source = DataSource(val_df, source.roots, val_folds)
        yield train_source, val_source


def get_complimental_folds(folds: List[int]) -> List[int]:
    return list(set(range(8)) - set(folds))


def get_folds_by(
    source: DataSource, fold_index: int, n_fold: int
) -> Tuple[DataSource, DataSource]:
    return list(kfold_split(source, n_fold))[fold_index]
