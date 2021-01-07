import pandas as pd

from .base import BinaryBaseline, SurvivalBaseline


feature_types = [
    "firstorder",
    "shape",
    "glcm",
    "glszm",
    "glrlm",
    "gldm",
    "ngtdm"
]


class RadiomicsBaseline:
    def __init__(self, data_path, max_features_to_select=10, colnames=[]):
        self.data_path = data_path
        self.binary_model = BinaryBaseline(max_features_to_select=max_features_to_select)
        self.survival_model = SurvivalBaseline(max_features_to_select=max_features_to_select)
        self.colnames = colnames

        self.data_train, self.data_valid = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.data_path)
        if not self.colnames:
            columns = [c for c in data.columns if c not in ["Study ID", "split"]]
        elif isinstance(self.colnames, list):
            columns = self.colnames
        elif isinstance(self.colnames, str):
            columns = data.filter(regex=self.colnames)
        else:
            raise ValueError(f"Column names must be a list, str or None, got {self.colnames}.")

        for target in ["target_binary", "survival_time", "death"]:
            if target not in columns:
                columns.append(target)

        data_train, data_valid = data[data["split"] == "training"], data[data["split"] == "validation"]
        return data_train[columns], data_valid[columns]

    def _predict(self, target):
        X_train = self.data_train.drop(["target_binary", "survival_time", "death"], axis=1)
        X_valid = self.data_valid.drop(["target_binary", "survival_time", "death"], axis=1)
        if target == "binary":
            y_train = self.data_train.pop("target_binary")
            y_valid = self.data_valid.pop("target_binary")
            model = self.binary_model
        elif target == "survival":
            y_train = self.data_train.pop("survival_time")
            y_valid = self.data_valid.pop("survival_time"), data_valid.pop("death")
            model = self.survival_model
        model.train(X_train, y_train)
        y_pred = model.predict(X_valid)
        return y_pred

    def get_test_predictions(self):
        pred_class = self._predict("binary")
        pred_risk, pred_surv = self._predict("survival")

        results = {
            "binary": pred_class,
            "survival_risk": pred_risk,
            "survival_time": pred_surv
        }
        return results

