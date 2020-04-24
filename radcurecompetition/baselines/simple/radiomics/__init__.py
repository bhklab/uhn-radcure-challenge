import pandas as pd

from .base import BinaryBaseline, SurvivalBaseline


def load_data(path, target="binary"):
    data = pd.read_csv(path)
    data = data.drop([c for c in data.columns if "diagnostic" not in c] + ["Image", "Mask", "Study ID"], axis=1)
    if target == "binary":
        data = data.drop(["survival_time", "death"], axis=1)
    elif target == "survival":
        data = data.drop("target_binary", axis=1)
    else:
        raise ValueError(f"Target must be one of ['binary', 'survival'], got {target}.")

    data_train, data_valid = data[data["split"] == "training"], data[data["split"] == "validation"]
    return data_train.drop("split", axis=1), data_valid.drop("split", axis=1)


def run_binary(data_path="../../data/radiomics.csv"):
    data_train, data_valid = load_data(data_path, target="binary")
    X_train = data_train.drop("target_binary", axis=1)
    y_train = data_train.pop("target_binary")
    X_valid = data_valid.drop("target_binary", axis=1)
    y_valid = data_valid.pop("target_binary")
    # select features
    model = BinaryBaseline()
    model.train(X_train, y_train)
    y_pred = model.predict(X_valid)
    return y_valid, y_pred


def run_survival(data_path="../../data/radiomics.csv"):
    data_train, data_valid = load_data(data_path, target="survival")
    X_train = data_train.drop("survival_time", axis=1)
    y_train = data_train.pop("survival_time")
    X_valid = data_valid.drop(["survival_time", "death"], axis=1)
    y_valid = data_valid.pop("survival_time"), data_valid.pop("death")
    # select features
    model = SurvivalBaseline()
    model.train(X_train, y_train)
    y_pred = model.predict(X_valid)
    return y_valid, y_pred
