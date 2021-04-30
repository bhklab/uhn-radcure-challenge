"""Base classes for binary and survival benchmark models."""
from typing import Union, List, Tuple, Dict, Optional

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
from sklearn.preprocessing import StandardScaler


class SelectMRMRe(BaseEstimator):
    """Scikit-learn compatible wrapper around the pymrmre feature selection
    package.

    Notes
    -----
    See the pyMRMRe documentation <https://github.com/bhklab/PymRMRe> and the
    original publication [1]_ for more information.

    References
    ----------
    .. [1] Hanchuan Peng et al. ‘Feature selection based on mutual information
       criteria of max-dependency, max-relevance, and min-redundancy’,
       IEEE Trans. Pattern Anal. Machine Intell., vol. 27, no. 8,
       pp. 1226–1238, Aug. 2005
    """
    def __init__(self,
                 n_features: int = 10,
                 var_prefilter_thresh: float = 0.,
                 target_col: Optional[str] = None):
        """Initialize the class.

        Parameters
        ----------
        n_features
            The number of features to select.
        var_prefilter_thresh
            If > 0, features with variance < `var_prefilter_thresh` will be
            dropped before performing selection.
        target_col
            The name of the target column. If None, the targets must be passed
            to the `fit` method.
        """
        self.n_features = n_features
        self.var_prefilter_thresh = var_prefilter_thresh
        self.target_col = target_col

    def fit(self, X: pd.DataFrame,
            y: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None
            ) -> "SelectMRMRe":
        """Perform feature selection.

        The selected features will be saved in order to enable selection on
        new data using the `transform` method. Note that pyMRMRe expects a
        Pandas DataFrame in contrast with most scikit-learn estimators.

        Parameters
        ----------
        X
            The feature matrix.If `self.target_col` is not None, the dataframe
            should also contain the target column.
        y
            The target vector. Only used if `self.target_col` is None.

        Returns
        -------
        SelectMRMRe
            The fitted object.
        """
        X_select = X.copy()
        if not isinstance(X_select, pd.DataFrame):
            X_select = pd.DataFrame(X_select)
            if isinstance(y, (pd.DataFrame, pd.Series)):
                X_select = X_select.set_index(y.index)

        # filter out low variance features
        if self.var_prefilter_thresh > 0:
            X_select = X_select.loc[:, X_select.var(axis=0) > self.var_prefilter_thresh]

        if self.target_col is not None:
            target_col = self.target_col
            target_idx = X_select.columns.get_loc(target_col)
        else:
            X_select["target"] = y
            target_col = "target"
            target_idx = None

        # NOTE survival data is treated as continuous here, since pyMRMRe
        # v XXX does not seem to handle it properly yet.
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

    def transform(self, X: pd.DataFrame,
                  y: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None
                  ) -> pd.DataFrame:
        """Perform feature selection on a dataset.

        The features selected in the fit call are used. Note that pyMRMRe
        expects a Pandas DataFrame in contrast with most scikit-learn
        estimators.

        Parameters
        ----------
        X
            The feature matrix. If `self.target_col` is not None, the dataframe
            should also contain the target column.
        y
            The target vector. This argument is only included for sklearn
            pipeline compatibility and is otherwise ignored.

        Returns
        -------
        pd.DataFrame
            The selected features.
        """
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices_]
        else:
            return X[:, self.feature_indices_]

    def fit_transform(self, X: pd.DataFrame,
                      y: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None
                      ) -> pd.DataFrame:
        """Find optimal features to select and transform the dataset.

        This method simply applies fit and transform in sequence.

        Parameters
        ----------
        X
            The feature matrix. If `self.target_col` is not None, the dataframe
            should also contain the target column.
        y
            The target vector. This argument is only included for sklearn
            pipeline compatibility and is otherwise ignored.

        Returns
        -------
        pd.DataFrame
            The selected features.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def set_params(self,
                   n_features: Optional[int] = None,
                   var_prefilter_thresh: Optional[float] = None):
        """Set the estimator parameters.

        This is used by sklearn grid search and related hyperparameter tuning
        methods.

        Parameters
        ----------
        n_features
            The number of features to select.
        var_prefilter_thresh
            If > 0, features with variance < `var_prefilter_thresh` will be
            dropped before performing selection
        """
        if n_features is not None:
            self.n_features = n_features
        if var_prefilter_thresh is not None:
            self.var_prefilter_thresh = var_prefilter_thresh

    def __repr__(self):
        return (f"SelectMRMRe(n_features={self.n_features}, "
                f"var_prefilter_thresh={self.var_prefilter_thresh})")


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
                                          make_column_selector(dtype_include=np.floating))],
                                        remainder="passthrough")
        logistic = LogisticRegressionCV(class_weight="balanced",
                                        scoring="roc_auc",
                                        solver="saga",
                                        max_iter=5000,
                                        n_jobs=self.n_jobs)

        if self.max_features_to_select > 0:
            select = SelectMRMRe()
            pipe = make_pipeline(transformer, select, logistic)
            param_grid = {
                "selectmrmre__n_features": np.arange(2, self.max_features_to_select + 1)
            }
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
        return self.model.predict_proba(X)[:, 1]


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
                                                make_column_selector(dtype_include=np.floating))],
                                             remainder="passthrough")
        CoxRegression = sklearn_adapter(CoxPHFitter,
                                        event_col="death",
                                        predict_method="predict_partial_hazard")
        cox = CoxRegression()
        param_grid = {"sklearncoxphfitter__penalizer": 10.0**np.arange(-2, 3)}
        if self.max_features_to_select > 0:
            select = SelectMRMRe(target_col="death")
            # we can't put CoxRegression in the pipeline since sklearn
            # transformers cannot return data frames
            pipe = make_pipeline(select, cox)
            param_grid["selectmrmre__n_features"] = np.arange(2, self.max_features_to_select + 1)
        else:
            pipe = make_pipeline(cox)

        # NOTE lifelines sklearn adapter does not support parallelization
        self.model = GridSearchCV(pipe, param_grid, n_jobs=1)

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
        columns = X.columns.drop("death")
        X_transformed = self.transformer.fit_transform(X.drop("death", axis=1))
        X_transformed = pd.DataFrame(X_transformed, index=y.index, columns=columns)
        X_transformed["death"] = death
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X: pd.DataFrame, times: Union[np.ndarray, List[float], None] = None):
        """Generate predictions for new data.

        This method outputs the predicted partial hazard values for each row
        in `X`. The partial hazard is computed as
        :math: `\exp(X - \text{mean}(X_{\text{train}}\beta))` and corresponds
        to `type="risk"`in R's `coxph`. Additionally, it computes
        the predicted survival function for each subject in X.

        Parameters
        ----------
        X
            Prediction inputs with samples as rows and features as columns.
        times, optional
            Time bins to use for survival function prediction. If None
            (default), uses monthly bins up to 2 years.

        Returns
        -------
        np.ndarray
            Predictions for each row in `X`.

        """
        if times is None:
            # predict risk every month up to 2 years
            times = np.linspace(1, 2, 23)

        death = X["death"]
        columns = X.columns.drop("death")
        X_transformed = self.transformer.transform(X.drop("death", axis=1))
        X_transformed = pd.DataFrame(X_transformed, columns=columns)
        X_transformed["death"] = death

        # We need to change the predict method of the lifelines model
        # while still running the whole pipeline
        setattr(self.model.best_estimator_["sklearncoxphfitter"],
                "_predict_method", "predict_survival_function")
        # GridSearchCV.predict does not support keyword arguments
        pred_surv = self.model.best_estimator_.predict(X_transformed, times=times).T
        setattr(self.model.best_estimator_["sklearncoxphfitter"],
                "_predict_method", "predict_partial_hazard")
        pred_risk = self.model.predict(X_transformed)
        return pred_risk, pred_surv


class SimpleBaseline:
    """Convenience class to train simple binary and survival baselines.

    This class handles splitting the data, model training and prediction.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 max_features_to_select: int = 10,
                 colnames: List[str] = [],
                 n_jobs: int = -1):
        """Initialize the class.

        Parameters
        ----------
        data
            The dataset used for training and prediction.
        max_features_to_select
            How many features to select. If 0, no feature selection will
            be performed.
        colnames
            List of columns to use as input features (must be present in `data`)
        n_jobs
            Number of parallel processes to use.
        """
        self.binary_model = BinaryModel(
            max_features_to_select=max_features_to_select, n_jobs=n_jobs)
        self.survival_model = SurvivalModel(
            max_features_to_select=max_features_to_select, n_jobs=n_jobs)
        self.colnames = colnames

        self.data_train, self.data_test = self.prepare_data(data)
        if len(self.data_test) == 0:
            raise RuntimeError(("The test set is not available at this stage of"
                                " the challenge. You will be able to run and"
                                " evaluate the baseline models after the test"
                                " set is released."))

    def prepare_data(self,
                     data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test subsets, and select the correct
        columns.

        Parameters
        ----------
        data
            The dataset used for training and prediction.

        Returns
        -------
        tuple of pd.DataFrame
            The processed training and test datasets.
        """
        if not self.colnames:
            columns = [c for c in data.columns if c not in ["Study ID", "split"]]
        elif isinstance(self.colnames, list):
            columns = self.colnames
        elif isinstance(self.colnames, str):
            columns = data.filter(regex=self.colnames).columns.tolist()
        else:
            raise ValueError(("Column names must be a list, str or None,"
                              f" got {self.colnames}."))

        for target in ["target_binary", "survival_time", "death"]:
            if target not in columns:
                columns.append(target)

        data_train, data_test = data[data["split"] == "training"], data[data["split"] == "test"]
        return data_train[columns], data_test[columns]

    def _get_selected_feature_names(self, X_train, model):
        if isinstance(model.model, GridSearchCV):
            estimator = model.model.best_estimator_
        else:
            estimator = model.model
        if "selectmrmre" in estimator:
            feature_names = X_train.columns[estimator["selectmrmre"].feature_indices_].tolist()
        else:
            feature_names = X_train.columns.tolist()
        return feature_names

    def _train_and_predict(self,
                           target: str
                           ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """Train the model on a given task and return the test set predictions.

        Parameters
        ----------
        target
            If 'binary', train and predict on the binary task. If 'survival',
            train and predict on the survival task.

        Returns
        -------
        pd.Series or tuple of pd.Series
            If `target == 'binary'`, returns the predicted positive class
            probability for each subject in the test set. If
            `target == 'survival'`, returns the predicted risk score and
            survival function for each subject.
        """
        X_train = self.data_train.drop(["target_binary", "survival_time"], axis=1)
        X_test = self.data_test.drop(["target_binary", "survival_time"], axis=1)
        if target == "binary":
            X_train, X_test = X_train.drop("death", axis=1), X_test.drop("death", axis=1)
            y_train = self.data_train["target_binary"]
            model = self.binary_model
        elif target == "survival":
            y_train = self.data_train["survival_time"]
            model = self.survival_model

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        setattr(self, f"selected_features_{target}_",
                self._get_selected_feature_names(X_train, model))

        return y_pred

    def get_test_predictions(self) -> Dict[str, pd.Series]:
        """Train the model on binary and survival tasks and return the test
        set predictions.

        Returns
        -------
        dict
            The test set predictions for binary classification, survival
            risk score and survival function.
        """
        pred_binary = self._train_and_predict("binary")
        pred_event, pred_time = self._train_and_predict("survival")

        pred = {
            "binary": pred_binary,
            "survival_event": pred_event,
            "survival_time": pred_time
        }
        return pred
