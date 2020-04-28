from typing import Union, List

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter

from pymrmre import mrmr_ensemble

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SelectMRMR(BaseEstimator):
    def __init__(self, n_features=10, var_prefilter_thresh=0., target_col=None):
        self.n_features = n_features
        self.var_prefilter_thresh = var_prefilter_thresh
        self.target_col = target_col

    def fit(self, X, y=None):
        X_select = X.copy()
        if not isinstance(X_select, pd.DataFrame):
            X_select = pd.DataFrame(X_select)
            if isinstance(y, (pd.DataFrame, pd.Series)):
                X_select = X_select.set_index(y.index)

        if self.var_prefilter_thresh > 0:
            X_select = X_select.loc[:, X_select.var(axis=0) > self.var_prefilter_thresh]

        if self.target_col is not None:
            target_col = self.target_col
            target_idx = X_select.columns.get_loc(target_col)
        else:
            X_select["target"] = y
            target_col = "target"
            target_idx = None

        if np.issubdtype(y.dtype, np.integer):
            target_type = 1
        else:
            target_type = 0
        selected = mrmr_ensemble(X_select,
                                 [target_col],
                                 [0]*(len(X_select.columns) - 1) + [target_type],
                                 self.n_features,
                                 1,
                                 0,
                                 return_index=True)
        selected = selected.tolist()[0][0]
        self.feature_indices_ = selected
        if target_idx:
            self.feature_indices_.append(target_idx)
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices_]
        else:
            return X[:, self.feature_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def set_params(self, n_features=None, var_prefilter_thresh=None):
        if n_features is not None:
            self.n_features = n_features
        if var_prefilter_thresh is not None:
            self.var_prefilter_thresh = var_prefilter_thresh

    def __repr__(self):
        return f"SelectMRMR(n_features={self.n_features}, var_prefilter_thresh={self.var_prefilter_thresh})"


class BinaryModel:
    """Baseline model for binary classification task.

    This class uses penalized binary logistic regression as the estimator. The
    L2 penalty coefficient is tuned using cross validation.
    """

    def __init__(self, max_features_to_select=0, n_jobs: int = -1):
        """Initialize the class.

        Parameters
        ----------
        n_jobs
            Number of parallel processes to use for cross-validation.
            If `n_jobs == -1` (default), use all available CPUs.
        """
        self.max_features_to_select = max_features_to_select
        self.n_jobs = n_jobs
        transformer = ColumnTransformer([('scale', StandardScaler(),
                                          make_column_selector(dtype_include=np.number)),
                                         ('onehot',
                                          OneHotEncoder(drop="first", sparse=False),
                                          make_column_selector(dtype_include=object))])
        logistic = LogisticRegressionCV(class_weight="balanced",
                                        scoring="roc_auc",
                                        solver="saga",
                                        n_jobs=self.n_jobs)
        if self.max_features_to_select > 0:
            select = SelectMRMR()
            pipe = make_pipeline(transformer, select, logistic)
            param_grid = {"selectmrmr__n_features": np.arange(2, self.max_features_to_select + 1)}
            self.model = GridSearchCV(pipe, param_grid, n_jobs=self.n_jobs)
        else:
            self.model = make_pipeline(transformer, logistic)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BinaryModel":
        """Train the model.

        This method returns `self` to make method chaining easier.

        Parameters
        ----------
        X
            The feature matrix with samples in rows and features in columns.

        y
            The prediction targets encoded as integers, with 1 denoting
            the positive class.

        Returns
        -------
        BinaryBaseline
            The trained model.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data.

        This method outputs the predicted positive class probabilities for
        each row in `X`.

        Parameters
        ----------
        X
            Prediction inputs with samples as rows and features as columns.

        Returns
        -------
        np.ndarray
            Predicted positive class probabilities for each row in `X`.

        """
        return self.model.predict_proba(X)


class SurvivalModel:
    """Baseline model for survival prediction task.

    This class uses penalized Cox proportional hazard regression as the
    estimator. The L2 penalty coefficient is tuned using cross validation.
    """

    def __init__(self, max_features_to_select=0, n_jobs: int = -1):
        """Initialize the class.

        Parameters
        ----------
        n_jobs
            Number of parallel processes to use for cross-validation.
            If `n_jobs == -1` (default), use all available CPUs.
        """
        self.max_features_to_select = max_features_to_select
        self.n_jobs = n_jobs
        self.transformer = ColumnTransformer([('scale', StandardScaler(),
                                               make_column_selector(dtype_include=np.number)),
                                              ('onehot',
                                               OneHotEncoder(drop="first", sparse=False),
                                               make_column_selector(dtype_include=object))])
        CoxRegression = sklearn_adapter(CoxPHFitter,
                                        event_col="death",
                                        predict_method="predict_partial_hazard")
        param_grid = {"sklearncoxphfitter__penalizer": 10.0**np.arange(-2, 3)}
        if self.max_features_to_select > 0:
            select = SelectMRMR(target_col="death")
            # can't put CoxRegression in the pipeline since sklearn
            # transformers cannot return data frames
            pipe = make_pipeline(select, CoxRegression())
            param_grid["selectmrmr__n_features"] = np.arange(2, self.max_features_to_select + 1)
        else:
            pipe = make_pipeline(CoxRegression())
        self.model = GridSearchCV(pipe, param_grid, n_jobs=self.n_jobs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "SurvivalModel":
        """Train the model.

        This method returns `self` to make method chaining easier. Note that
        event information is passed as a column in `X`, while `y` contains
        the time-to-event data.

        Parameters
        ----------
        X
            The feature matrix with samples in rows and features in columns.
            It should also contain a column named 'death' with events encoded
            as integers (1=event, 0=censoring).

        y
            Times to event or censoring.

        Returns
        -------
        SurvivalBaseline
            The trained model.
        """
        death = X["death"]
        X = X.drop("death", axis=1)
        X = self.transformer.fit_transform(X)
        X = pd.DataFrame(X, index=y.index)
        X["death"] = death
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame, times: Union[np.ndarray, List[float], None] = None):
        """Generate predictions for new data.

        This method outputs the predicted partial hazard values for each row
        in `X`. The partial hazard is computed as
        :math: `-\exp(X - \text{mean}(X_{\text{train}}\beta))` and corresponds
        to `type="risk"`in R's `coxph`. Additionally, it computes
        the predicted survival function for each subject in X.

        Parameters
        ----------
        X
            Prediction inputs with samples as rows and features as columns.
        times, optional
            Time bins to use for survival function prediction. If None (default),
            uses monthly bins up to 2 years.

        Returns
        -------
        np.ndarray
            Predictions for each row in `X`.

        """
        if times is None:
            # predict risk every month up to 2 years
            times = np.linspace(1, 2, 23)

        death = X["death"]
        X = X.drop("death", axis=1)
        X = self.transformer.transform(X)
        X = pd.DataFrame(X)
        X["death"] = death
        # We need to change the predict method of the lifelines model
        # while still running the whole pipeline.
        # This is a somewhat ugly hack, there might be a better way to do it.
        setattr(self.model.best_estimator_.named_steps["sklearncoxphfitter"], "_predict_method", "predict_survival_function")
        # GridSearchCV.predict does not support keyword arguments
        pred_surv = self.model.best_estimator_.predict(X, times=times).T
        setattr(self.model.best_estimator_.named_steps["sklearncoxphfitter"], "_predict_method", "predict_partial_hazard")
        pred_risk = -self.model.predict(X)
        return pred_risk, pred_surv


class SimpleBaseline:
    def __init__(self, data, max_features_to_select=10, colnames=[]):
        self.binary_model = BinaryModel(max_features_to_select=max_features_to_select)
        self.survival_model = SurvivalModel(max_features_to_select=max_features_to_select)
        self.colnames = colnames

        self.data_train, self.data_valid = self.prepare_data(data)

    def prepare_data(self, data):
        if not self.colnames:
            columns = [c for c in data.columns if c not in ["Study ID", "split"]]
        elif isinstance(self.colnames, list):
            columns = self.colnames
        elif isinstance(self.colnames, str):
            columns = data.filter(regex=self.colnames).columns.tolist()
        else:
            raise ValueError(f"Column names must be a list, str or None, got {self.colnames}.")

        for target in ["target_binary", "survival_time", "death"]:
            if target not in columns:
                columns.append(target)

        data = data.fillna(-999)

        data_train, data_valid = data[data["split"] == "training"], data[data["split"] == "validation"]
        return data_train[columns], data_valid[columns]

    def _predict(self, target):
        X_train = self.data_train.drop(["target_binary", "survival_time"], axis=1)
        X_valid = self.data_valid.drop(["target_binary", "survival_time"], axis=1)
        if target == "binary":
            X_train, X_valid = X_train.drop("death", axis=1), X_valid.drop("death", axis=1)
            y_train = self.data_train["target_binary"]
            y_valid = self.data_valid["target_binary"]
            model = self.binary_model
        elif target == "survival":
            y_train = self.data_train["survival_time"]
            y_valid = self.data_valid["death"], self.data_valid["survival_time"]
            model = self.survival_model

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return y_valid, y_pred

    def get_test_predictions(self, times=None):
        true_binary, pred_binary = self._predict("binary")
        (true_event, true_time), (pred_event, pred_time) = self._predict("survival")

        true = {
            "binary": true_binary,
            "survival_event": true_event,
            "survival_time": true_time
        }
        pred = {
            "binary": pred_binary,
            "survival_event": pred_event,
            "survival_time": pred_time
        }
        return true, pred
