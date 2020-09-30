from pathlib import Path
from typing import Tuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tqdm import tqdm

from .datasource import DataSource
from .util import get_osic_pulmonary_fibrosis_progression_root


def _prepare_folds(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    target = train_df["FVC"]
    groups = train_df["Patient"].values
    for n, (train_index, val_index) in enumerate(
        GroupKFold(n_splits=8).split(train_df, target, groups)
    ):
        train_df.loc[val_index, "fold"] = int(n)
    train_df["fold"] = train_df["fold"].astype(int)
    return train_df


def load_osic_pulmonary_fibrosis_progression_csv(
    use_pseudo_baselines: bool = False, ignore_bad_ct: bool = False
):
    (
        train_df,
        test_df,
        submission_df,
    ) = load_osic_pulmonary_fibrosis_progression_csv_dataframe()

    train_df["Patient_Week"] = (
        train_df["Patient"].astype(str) + "_" + train_df["Weeks"].astype(str)
    )

    output = pd.DataFrame()
    gb = train_df.groupby("Patient")
    tk0 = tqdm(gb, total=len(gb))
    for _, usr_df in tk0:
        rename_cols = {
            "Percent": "base_Percent",
            "Age": "base_Age",
        }
        if use_pseudo_baselines:
            usr_output = pd.DataFrame()
            for _, tmp in usr_df.groupby("Weeks"):
                tmp = tmp.drop(columns="Patient_Week").rename(
                    columns={**rename_cols, "Weeks": "base_Week", "FVC": "base_FVC"}
                )
                drop_cols = ["Age", "Sex", "SmokingStatus", "Percent"]
                _usr_output = (
                    usr_df.drop(columns=drop_cols)
                    .rename(columns={"Weeks": "predict_Week"})
                    .merge(tmp, on="Patient")
                )
                _usr_output["Week_passed"] = (
                    _usr_output["predict_Week"] - _usr_output["base_Week"]
                )
                usr_output = pd.concat([usr_output, _usr_output])
        else:
            usr_output = usr_df.rename(columns={**rename_cols, "Weeks": "predict_Week"})
            usr_output["base_FVC"] = usr_output["FVC"]
            usr_output["base_Week"] = usr_output["predict_Week"].min()
            usr_output["Week_passed"] = (
                usr_output["predict_Week"] - usr_output["base_Week"]
            )
        output = pd.concat([output, usr_output])
    train_df = output[output["Week_passed"] != 0].reset_index(drop=True)

    train_df = cast(pd.DataFrame, train_df)
    train_df = _prepare_folds(train_df)

    test_df = test_df.rename(
        columns={
            "Weeks": "base_Week",
            "FVC": "base_FVC",
            "Percent": "base_Percent",
            "Age": "base_Age",
        }
    )
    submission_df["Patient"] = submission_df["Patient_Week"].apply(
        lambda x: x.split("_")[0]
    )
    submission_df["predict_Week"] = (
        submission_df["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
    )
    test_df = submission_df.drop(columns=["FVC", "Confidence"]).merge(
        test_df, on="Patient"
    )
    test_df["Week_passed"] = test_df["predict_Week"] - test_df["base_Week"]

    train_img_roots = get_osic_pulmonary_fibrosis_progression_root() / "train"
    test_img_roots = get_osic_pulmonary_fibrosis_progression_root() / "test"

    train_folds = list(train_df["fold"].unique())
    test_folds = []

    train_df["Sex"] = train_df["Sex"].map({"Male": 1, "Female": 0})
    test_df["Sex"] = test_df["Sex"].map({"Male": 1, "Female": 0})

    concat = pd.concat(
        [train_df["SmokingStatus"], test_df["SmokingStatus"]], ignore_index=True
    )
    dummies = pd.get_dummies(concat, dtype=np.uint8, prefix="SmokingStatus")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat(
        [test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1
    )
    train_df["SmokingStatus"] = train_df["SmokingStatus"].map(
        {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
    )
    test_df["SmokingStatus"] = test_df["SmokingStatus"].map(
        {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
    )

    if ignore_bad_ct:
        train_df = train_df.drop(
            train_df[train_df.Patient == "ID00011637202177653955184"].index
        )
        train_df = train_df.drop(
            train_df[train_df.Patient == "ID00052637202186188008618"].index
        )

    return (
        DataSource(train_df, train_img_roots, train_folds),
        DataSource(test_df, test_img_roots, test_folds),
    )

    return train_df, test_df


def load_osic_pulmonary_fibrosis_progression_csv_dataframe() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    dataset_root_path = get_osic_pulmonary_fibrosis_progression_root()

    train_csv_path = dataset_root_path / "train.csv"
    train_df = cast(pd.DataFrame, pd.read_csv(train_csv_path))

    test_csv_path = dataset_root_path / "test.csv"
    test_df = cast(pd.DataFrame, pd.read_csv(test_csv_path))

    submission_csv_path = dataset_root_path / "sample_submission.csv"
    submission_df = cast(pd.DataFrame, pd.read_csv(submission_csv_path))

    return train_df, test_df, submission_df


def save_result(result: DataSource, dst_path: Path = Path("./submission.csv")) -> None:
    submission_csv_path = (
        get_osic_pulmonary_fibrosis_progression_root() / "sample_submission.csv"
    )
    submission_df = cast(pd.DataFrame, pd.read_csv(submission_csv_path))

    sub = submission_df.drop(columns=["FVC", "Confidence"]).merge(
        result.df[["Patient_Week", "FVC_pred", "Confidence"]], on="Patient_Week"
    )
    sub.columns = submission_df.columns
    sub.to_csv("submission.csv", index=False)
