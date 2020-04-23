import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter


class BaselineModel:
    def train(self, X, y):
        """Optimize model parameters using inputs `X` and targets `y` and returns
        the trained model (`self`)."""
        raise NotImplementedError

    def predict(self, X):
        """Generate predictions for new data."""
        raise NotImplementedError


class BinaryBaseline(BaselineModel):
    """Baseline model for binary classification task.

    This class uses penalized binary logistic regression as the estimator. The
    L2 penalty coefficient is tuned using cross validation.
    """
    def __init__(self, n_jobs: int = -1):
        """Initialize the class.

        Parameters
        ----------
        n_jobs
            Number of parallel processes to use for cross-validation.
            If `n_jobs == -1` (default), use all available CPUs.
        """
        self.n_jobs = n_jobs
        self.model = make_pipeline(
            ColumnTransformer([('scale', StandardScaler(),
                                make_column_selector(dtype_include=np.number)),
                               ('onehot',
                                OneHotEncoder(drop="first", sparse=False),
                                make_column_selector(dtype_include=object))]),
            LogisticRegressionCV(class_weight="balanced",
                                 scoring="roc_auc",
                                 n_jobs=self.n_jobs))

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> "BinaryBaseline":
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


class SurvivalBaseline(BaselineModel):
    """Baseline model for survival prediction task.

    This class uses penalized Cox proportional hazard regression as the
    estimator. The L2 penalty coefficient is tuned using cross validation.
    """
    def __init__(self, n_jobs: int = -1):
        """Initialize the class.

        Parameters
        ----------
        n_jobs
            Number of parallel processes to use for cross-validation.
            If `n_jobs == -1` (default), use all available CPUs.
        """
        self.n_jobs = n_jobs
        CoxRegression = sklearn_adapter(CoxPHFitter,
                                        event_col="death",
                                        predict_method="predict_partial_hazard")
        pipe = make_pipeline(
            ColumnTransformer([('scale', StandardScaler(),
                                make_column_selector(dtype_include=np.number)),
                               ('onehot',
                                OneHotEncoder(drop="first", sparse=False),
                                make_column_selector(dtype_include=object))]),
            CoxRegression())
        self.model = GridSearchCV(
            pipe,
            param_grid={"coxph__penalizer": 10.0**np.arange(-2, 3)},
            n_jobs=self.n_jobs)

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> "SurvivalBaseline":
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
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        """Generate predictions for new data.

        This method outputs the predicted partial hazard values for each row
        in `X`. The partial hazard is computed as
        :math: `\exp(X - \text{mean}(X_{\text{train}}\beta))` and corresponds
        to `type="risk"`in R's `coxph`.

        Parameters
        ----------
        X
            Prediction inputs with samples as rows and features as columns.

        Returns
        -------
        np.ndarray
            Predicted partial hazard for each row in `X`.

        """
        return self.model.predict(X)
