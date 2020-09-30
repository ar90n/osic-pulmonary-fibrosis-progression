import sys
from dataclasses import asdict
from typing import Optional, Any, List, Tuple, Iterable, Callable
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from torchtoolbox import transform

import pandas as pd
import numpy as np

from ..config import Config
from ..datasource import DataSource, train_validate_split, get_folds_by, kfold_split
from ..dataset import CTDataset
from ..lightgbm import train, infer
from ..transforms import DropSlice, LungMask, DescribeVolume


def get_original_label_of(oof: pd.DataFrame) -> str:
    return oof.columns[0][:-5]


def train_splits(
    config: Config,
    all_source: DataSource,
    transforms: Optional[Callable] = None,
    use_one_hot_encoding: bool = False,
    use_ct_data: bool = True,
):
    if transforms is None:
        transforms = transform.Compose([DropSlice(), LungMask(), DescribeVolume()])

    train_source, val_source = train_validate_split(all_source)
    train_dataset = CTDataset(
        train_source,
        train=True,
        transforms=transforms,
        use_one_hot_encoding=use_one_hot_encoding,
        use_ct_data=use_ct_data,
    )
    val_dataset = CTDataset(
        val_source,
        train=True,
        transforms=transforms,
        use_one_hot_encoding=use_one_hot_encoding,
        use_ct_data=use_ct_data,
    )
    model, oof = train(train_dataset, val_dataset, fold_index=1, n_fold=1)

    return model, oof


def train_nth_fold(
    config: Config,
    all_source: DataSource,
    fold_index: int,
    n_fold: int,
    transforms: Optional[Callable] = None,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
    use_one_hot_encoding: bool = False,
    use_ct_data: bool = True,
):
    if n_fold < 0 or 8 < n_fold:
        raise ValueError("n_fold must be best_model_path 1 with 8")

    train_source, val_source = get_folds_by(all_source, fold_index, n_fold)
    train_dataset = CTDataset(
        train_source,
        train=True,
        target=target,
        features=features,
        transforms=transforms,
        use_one_hot_encoding=use_one_hot_encoding,
        use_ct_data=use_ct_data,
    )
    val_dataset = CTDataset(
        val_source,
        train=True,
        target=target,
        features=features,
        transforms=transforms,
        use_one_hot_encoding=use_one_hot_encoding,
        use_ct_data=use_ct_data,
    )

    model, oof = train(train_dataset, val_dataset, fold_index=fold_index, n_fold=n_fold)

    return model, oof


def train_all_folds(
    config: Config,
    all_source: DataSource,
    n_fold: int = 8,
    n_workers: Optional[int] = 4,
    transforms: Optional[Callable] = None,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
    use_one_hot_encoding: bool = False,
    use_ct_data: bool = True,
) -> Tuple[Tuple[Any], pd.DataFrame]:
    if n_workers is None:
        n_workers = n_fold
    if n_fold < 0 or 8 < n_fold:
        raise ValueError("n_fold must be best_model_path 1 with 8")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for fold_index in range(n_fold):
            future = executor.submit(
                train_nth_fold,
                config,
                all_source,
                fold_index,
                n_fold,
                transforms,
                target,
                features,
                use_one_hot_encoding,
                use_ct_data,
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


def infer_with_all_folds(
    config: Config,
    test_source: DataSource,
    models: Iterable,
    transforms: Optional[Callable] = None,
    use_one_hot_encoding: bool = False,
    use_ct_data: bool = True,
):
    test_dataset = CTDataset(
        test_source,
        train=False,
        transforms=transforms,
        use_one_hot_encoding=use_one_hot_encoding,
        use_ct_data=use_ct_data,
    )
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
