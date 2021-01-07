import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve,
                             precision_recall_curve, auc)

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from joblib import Parallel, delayed
from typing import Tuple, Callable, Dict, Optional, Union, List

np.random.seed(42)

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
        cox = CoxRegression(step_size = 0.5)
        param_grid = {"sklearncoxphfitter__penalizer": 10.0**np.arange(-2, 3)}
        if self.max_features_to_select > 0:
            select = SelectMRMRe(target_col="death")
            # can't put CoxRegression in the pipeline since sklearn
            # transformers cannot return data frames
            pipe = make_pipeline(select, cox)
            param_grid["selectmrmre__n_features"] = np.arange(2, self.max_features_to_select + 1)
        else:
            pipe = make_pipeline(cox)

        # XXX lifelines sklearn adapter does not support parallelization
        # for now, need to find a better workaround
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
        columns = X.columns.drop("death")
        X_transformed = self.transformer.transform(X.drop("death", axis=1))
        X_transformed = pd.DataFrame(X_transformed, columns=columns)
        X_transformed["death"] = death

        # We need to change the predict method of the lifelines model
        # while still running the whole pipeline.
        # This is a somewhat ugly hack, there might be a better way to do it.
        setattr(self.model.best_estimator_["sklearncoxphfitter"],
                "_predict_method", "predict_survival_function")
        # GridSearchCV.predict does not support keyword arguments
        pred_surv = self.model.best_estimator_.predict(X_transformed, times=times).T
        setattr(self.model.best_estimator_["sklearncoxphfitter"],
                "_predict_method", "predict_partial_hazard")
        pred_risk = self.model.predict(X_transformed)
        return pred_risk, pred_surv
        
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
                                        solver="lbfgs",
                                        max_iter=1000,
                                        n_jobs=self.n_jobs)
        if self.max_features_to_select > 0:
            select = SelectMRMRe()
            pipe = make_pipeline(transformer, select, logistic)
            param_grid = {"selectmrmre__n_features": np.arange(2, self.max_features_to_select + 1)}
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

class SimpleBaseline:
    """Convenience class to train simple binary and survival baselines.

    This class handles splitting the data, model training and prediction.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 fuzzy_feature: List[str] = [],
                 max_features_to_select: int = 10,
                 colnames_fuzzy: List[str] = [],
                 pos_drop_colnames: List[str] = [],
                 n_jobs: int = -1):
        """Initialize the class.

        Parameters
        ----------
        data
            The dataset used for training and prediction.
        fuzzy_feature
             Feature to be used for fuzzy split
        max_features_to_select
            How many features to select. If 0, no feature selection will
            be performed.
        colnames_fuzzy
            List of columns to use as input features for fuzzy model(must be present in `data`)
        n_jobs
            Number of parallel processes to use.
        """
        self.fuzzy_classifier = BinaryModel(max_features_to_select=0, n_jobs=n_jobs)

        self.low_binary_model = BinaryModel(max_features_to_select, n_jobs=n_jobs)
        self.high_binary_model = BinaryModel(max_features_to_select, n_jobs=n_jobs)

        self.low_survival_model = SurvivalModel(max_features_to_select, n_jobs=n_jobs)
        self.high_survival_model = SurvivalModel(max_features_to_select, n_jobs=n_jobs)
        
        
        self.colnames_fuzzy = colnames_fuzzy
        
        self.fuzzy_feature = fuzzy_feature

        self.data_train_fuzzy, self.data_test_fuzzy = self.prepare_data(data, self.colnames_fuzzy)

    def prepare_data(self,
                     data: pd.DataFrame, colnames: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        if not colnames:
            columns = [c for c in data.columns if c not in ["Study ID", "split"]]
        elif isinstance(colnames, list):
            columns = colnames
        elif isinstance(self.clin_colnames, str):
            columns = data.filter(regex=self.colnames).columns.tolist()
        else:
            raise ValueError(("Column names must be a list, str or None,"
                              f" got {self.colnames}."))

        for target in ["target_binary", "survival_time", "death"]:
            if target not in columns:
                columns.append(target)


        data_train, data_test = data[data["split"] == "training"], data[data["split"] == "training"]
        train_columns = columns.copy()

        if self.fuzzy_feature:
            data_train = self.fuzzySplit(data_train)
            train_columns.append("fuzzy_binary")

        data_test = data_test[columns]
        data_train = data_train[train_columns]
        
        return data_train, data_test

    def fuzzySplit(self, data: pd.DataFrame) -> pd.DataFrame:
        """Binarize tumour volume to greater than or less than the median.
            Used to perform fitting of fuzzy model. 
            
        """
        median = data['original_shape_MeshVolume'].median()
        data.loc[data.original_shape_MeshVolume >= median, 'fuzzy_binary'] = 1
        data.loc[data.original_shape_MeshVolume < median, 'fuzzy_binary'] = 0

        return data

    def fuzzy_train(self, model, X_train, y_train, X_test, model_prob) -> np.array:
        """Train a model for the high or low volume subset of patients.
        
        Parameters
        ----------
        model
            the binary or survival model to be trained based on subset of patients based on tumour volume. 
        X_train
            training features
        y_train
            training outcome
        X_test
            testing features       
        model_prob
            the probability that a given test patient belongs to the given model
        
        Returns
        -------
        y_pred
            probability of a test patient having a certain outcome     
        """
        model.fit(X_train, y_train)
        X_test = pd.DataFrame(X_test, columns=X_train.columns)
        y_pred = model.predict(X_test)[:, 1] * model_prob
        return y_pred
        
    def _train_models(self, target: str) -> Tuple[Union[pd.Series, Tuple[pd.Series, pd.Series]]]:
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

        X_train_fuzzy = self.data_train_fuzzy.drop(["target_binary", "survival_time"], axis=1)
        X_test_fuzzy = self.data_test_fuzzy.drop(["target_binary", "survival_time"], axis=1)

        X_train_low, X_train_high, low_model_prob, high_model_prob = self.fuzzy_fit(self.fuzzy_classifier, X_train_fuzzy, X_test_fuzzy)

        if target == "binary":
            X_train_low, X_train_high, X_test_fuzzy = X_train_low.drop(["death"], axis=1), X_train_high.drop(["death"], axis=1), X_test_fuzzy.drop(["death"], axis=1)
            # X_train_low, X_train_high, X_test_fuzzy = X_train_low.drop(["death", 'original_shape_MeshVolume'], axis=1), X_train_high.drop(["death", 'original_shape_MeshVolume'], axis=1), X_test_fuzzy.drop(["death", 'original_shape_MeshVolume'], axis=1)
            y_train_low = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_low.index)]["target_binary"]
            y_train_high = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_high.index)]["target_binary"]
            
            #Fit and predict low/small tumour volume model
            low_y_pred = self.fuzzy_train(self.low_binary_model, X_train_low, y_train_low, X_test_fuzzy, low_model_prob)            
            #Fit and predict high/large tumour volume model
            high_y_pred = self.fuzzy_train(self.high_binary_model, X_train_high, y_train_high, X_test_fuzzy, high_model_prob)           
            pred = low_y_pred + high_y_pred
            
        elif target == "survival":
            X_train_low = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_low.index)].drop(["target_binary", "fuzzy_binary", "survival_time"], axis=1)
            X_train_high = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_high.index)].drop(["target_binary", "fuzzy_binary", "survival_time"], axis=1)
            # X_train_low = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_low.index)].drop(["target_binary", "fuzzy_binary", "survival_time", 'original_shape_MeshVolume'], axis=1)
            # X_train_high = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_high.index)].drop(["target_binary", "fuzzy_binary", "survival_time", 'original_shape_MeshVolume'], axis=1)
            y_train_low = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_low.index)]["survival_time"]
            y_train_high = self.data_train_fuzzy.loc[self.data_train_fuzzy.index.isin(X_train_high.index)]["survival_time"]
            
            # X_test_fuzzy = X_test_fuzzy.drop(['original_shape_MeshVolume'], axis=1)
            
            low_model = self.low_survival_model
            high_model = self.high_survival_model
            
            #Fit and predict low/small tumour volume model
            low_model.fit(X_train_low, y_train_low)
            low_y_pred_event, low_y_pred_time = low_model.predict(X_test_fuzzy)
            low_y_pred_event = low_y_pred_event * low_model_prob
            low_y_pred_time = low_y_pred_time * np.tile(low_model_prob.reshape(low_model_prob.shape[0],1),23)

            #Fit and predict high/large tumour volume model
            high_model.fit(X_train_high, y_train_high)
            high_y_pred_event, high_y_pred_time = high_model.predict(X_test_fuzzy)
            high_y_pred_event = high_y_pred_event * high_model_prob
            high_y_pred_time = high_y_pred_time * np.tile(high_model_prob.reshape(high_model_prob.shape[0],1),23)
            
            pred = (low_y_pred_event + high_y_pred_event), (low_y_pred_time + high_y_pred_time)        
            return (low_y_pred_event + high_y_pred_event) , (low_y_pred_time + high_y_pred_time)

        return pred


    def fuzzy_fit(self, model, X_train, X_test) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]:
        """ Train a logistic model to assign a probability of a patient having a large or small 
        (high or low, respectively) tumour volume.
        
            Parameters
            ----------
            model
                the binary model to be trained
            X_train
                training data 
            X_test
                test data

            Returns
            -------
            X_train_low and X_train_high
                training data will fuzzy feature removed
            low_model_prob and high_model_prob
                probability of a test patient having a large or small tumour
        """
        fuzzy_model = model
        
        # define the X and y training data
        fuzzy_X_train = pd.DataFrame(self.data_train_fuzzy[self.fuzzy_feature])
        fuzzy_y_train = self.data_train_fuzzy["fuzzy_binary"]

        #Fit and predict probability of belonging to low or high volume model
        fuzzy_model.fit(fuzzy_X_train, fuzzy_y_train)
        fuzzy_pred_train = fuzzy_model.predict(fuzzy_X_train)
        
        #Predict probabilities on testing data
        fuzzy_pred_test = fuzzy_model.predict(pd.DataFrame(X_test[self.fuzzy_feature]))
        low_model_prob = fuzzy_pred_test[:, 0] #probability of having small tumour
        high_model_prob = fuzzy_pred_test[:, 1] #probability of having large tumour

        fuzzy_train_pred_df = pd.DataFrame(fuzzy_pred_train, index=fuzzy_X_train.index)
        X_train_temp = pd.concat([fuzzy_train_pred_df, X_train], axis=1, join='inner')

        #Remove fuzzy feature from training data
        X_train_low = X_train_temp.loc[X_train_temp[0] >= 0.5]
        X_train_high = X_train_temp.loc[X_train_temp[0] < 0.5]
        to_drop = ["fuzzy_binary", 0, 1]
        X_train_low = X_train_low.drop(labels = to_drop, axis=1)
        X_train_high = X_train_high.drop(labels = to_drop, axis=1)

        return X_train_low, X_train_high, low_model_prob, high_model_prob
        
    def _train_and_predict(self,
                           target: str
                           ) -> Tuple[Union[pd.Series, Tuple[pd.Series, pd.Series]]]:

        """Call function to train the model on a given task and return the test set predictions.

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
        if target == "binary":
            pred_binary = self._train_models(target)
            return pred_binary
            
        elif target == "survival":
            pred_survival, pred_risk = self._train_models(target)
            return pred_survival, pred_risk

    def get_test_predictions(self) -> Dict[str, pd.Series]:
        """call functions to train the model on binary and survival tasks and return the test
        set predictions.

        Returns
        -------
        dict
            The test set predictions for binary classification, survival
            risk score and survival function.
        """
        pred_risk, pred_survival = self._train_and_predict("survival")
        pred_binary = self._train_and_predict("binary")
        
        pred = {
            "binary": pred_binary,
            "survival_event": pred_risk,
            "survival": pred_survival
        }
        return pred  
