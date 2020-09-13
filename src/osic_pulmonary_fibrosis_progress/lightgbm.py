from typing import List, Optional, Dict, Any

import lightgbm as lgb
import pandas as pd

from .datasource import DataSource


DEFAULT_PARAM = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "seed": 42,
    "max_depth": -1,
    "verbosity": -1,
}


def get_numerical_features(source: DataSource, target: str = "FVC") -> List[str]:
    cat_features = ["Sex", "SmokingStatus"]
    num_features = [
        c
        for c in source.df.columns
        if (source.df.dtypes[c] != "object") & (c not in cat_features)
    ]
    features = num_features
    drop_features = ["Patient_Week", target, "predict_Week", "base_Week"]
    return [c for c in features if c not in drop_features]


def run_single_lightgbm(
    train_source: DataSource,
    val_source: DataSource,
    param: Optional[Dict[str, Any]] = None,
    target: str = "FVC",
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
):
    if param is None:
        param = DEFAULT_PARAM
    if categorical_features is None:
        categorical_features = ["Sex", "SmokingStatus"]
    if numerical_features is None:
        numerical_features = get_numerical_features(train_source, target)
    features= categorical_features + numerical_features

    if categorical_features == []:
        train_data = lgb.Dataset(
            train_source.df[features], label=train_source.df[target]
        )
        val_data = lgb.Dataset(val_source.df[features], label=val_source.df[target])
    else:
        train_data = lgb.Dataset(
            train_source.df[features],
            label=train_source.df[target],
            categorical_feature=categorical_features,
        )
        val_data = lgb.Dataset(
            val_source.df[features],
            label=val_source.df[target],
            categorical_feature=categorical_features,
        )

    num_round = 10000
    model = lgb.train(
        param,
        train_data,
        num_round,
        valid_sets=[train_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=100,
    )
    return model
