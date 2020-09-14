import sys
from dataclasses import asdict
from typing import Optional, Any, List, Tuple, Iterable
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from ..config import Config
from ..datasource import DataSource, train_validate_split, get_folds_by, kfold_split
from ..dataset import TabularDataset
from ..lightgbm import train, infer


def get_original_label_of(oof: pd.DataFrame) -> str:
    return oof.columns[0][:-5]


def train_splits(config: Config, all_source: DataSource):
    train_source, val_source = train_validate_split(all_source)

    train_dataset = TabularDataset(train_source, train=True)
    val_dataset = TabularDataset(val_source, train=True)
    model, oof = train(train_dataset, val_dataset, fold_index=1, n_fold=1)

    return model, oof


def train_nth_fold(
    config: Config,
    all_source: DataSource,
    fold_index: int,
    n_fold: int,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
):
    if n_fold < 0 or 8 < n_fold:
        raise ValueError("n_fold must be best_model_path 1 with 8")

    train_source, val_source = get_folds_by(all_source, fold_index, n_fold)
    train_dataset = TabularDataset(
        train_source, train=True, target=target, features=features
    )
    val_dataset = TabularDataset(
        val_source, train=True, target=target, features=features
    )

    model, oof = train(train_dataset, val_dataset, fold_index=fold_index, n_fold=n_fold)

    return model, oof


def train_all_folds(
    config: Config,
    all_source: DataSource,
    n_fold: int = 8,
    n_workers: Optional[int] = 1,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
) -> Tuple[Tuple[Any], pd.DataFrame]:
    if n_workers is None:
        n_workers = n_fold
    if n_fold < 0 or 8 < n_fold:
        raise ValueError("n_fold must be best_model_path 1 with 8")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for fold_index in range(n_fold):
            future = executor.submit(
                train_nth_fold, config, all_source, fold_index, n_fold, target, features
            )
            futures.append(future)
    models, oofs = zip(*[f.result() for f in futures])
    oof = pd.concat(oofs).sort_index()
    if oof is None:
        raise RuntimeError("oof is None")

    # RMSE
    from sklearn.metrics import mean_squared_error

    target_label = get_original_label_of(oof)
    rmse = np.sqrt(
        mean_squared_error(all_source.df[target_label], np.squeeze(oof.values))
    )
    print(f"CV RMSE score: {rmse:<8.5f}")

    return models, oof


def infer_with_all_folds(config: Config, test_source: DataSource, models: Iterable):
    test_dataset = TabularDataset(test_source, train=False)
    avg = pd.concat((infer(m, test_dataset) for m in models), axis="columns").mean(
        axis="columns"
    )
    return pd.DataFrame({"prediction": avg})


def show_feature_importance(models: Iterable, features: List[str]):
    feature_importance_df = pd.DataFrame()
    for fold, m in enumerate(models):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = m.feature_importance(importance_type="gain")
        fold_importance_df["fold"] = fold
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0
        )

    cols = (
        feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:50]
        .index
    )
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    import matplotlib.pyplot as plt
    import seaborn as sns

    # plt.figure(figsize=(8, 16))
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x="importance",
        y="Feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("Features importance (averaged/folds)")
    plt.tight_layout()
