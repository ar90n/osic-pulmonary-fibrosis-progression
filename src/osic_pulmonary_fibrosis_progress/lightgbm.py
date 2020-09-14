from typing import List, Optional, Dict, Any

import lightgbm as lgb
import pandas as pd
import numpy as np

from .dataset import Dataset

DEFAULT_PARAM = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "seed": 42,
    "max_depth": -1,
    "verbosity": -1,
}


def train(
    train_dataset: Dataset,
    val_dataset: Dataset,
    fold_index: int,
    n_fold: int,
    param: Optional[Dict[str, Any]] = None,
):
    if param is None:
        param = DEFAULT_PARAM

    x_train, y_train = [np.vstack(d) for d in zip(*train_dataset)]
    x_val, y_val = [np.vstack(d) for d in zip(*val_dataset)]
    train_data = lgb.Dataset(x_train, label=np.squeeze(y_train))
    val_data = lgb.Dataset(x_val, label=np.squeeze(y_val))

    num_round = 10000
    model = lgb.train(
        param,
        train_data,
        num_round,
        valid_sets=[train_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=100,
    )

    target_pred = model.predict(x_val, num_iteration=model.best_iteration)
    target_pred_label = f"{train_dataset.target}_pred"
    oof = pd.DataFrame(
        {target_pred_label: target_pred}, index=val_dataset.source.df.index
    )
    return model, oof


def infer(model, test_dataset: pd.DataFrame) -> pd.DataFrame:
    test_pred = model.predict(test_dataset)
    return pd.DataFrame({"prediction": test_pred}, index=test_dataset.source.df.index)
