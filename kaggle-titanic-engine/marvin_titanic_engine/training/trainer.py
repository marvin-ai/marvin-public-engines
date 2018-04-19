#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler, scale
# from sklearn.linear_model import LogisticRegression

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from sklearn import svm, neighbors, tree
        from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, scale
        from sklearn.linear_model import LogisticRegression

        print("\n\nStarting grid search using SVM!")

        # Create a classifier with the parameter candidates
        svm_grid = GridSearchCV(estimator=svm.SVC(), param_grid=params["svm"], n_jobs=-1)

        # Train the classifier on training data
        svm_grid.fit(
            self.marvin_dataset['X_train'],
            self.marvin_dataset['y_train']
        )

        print("Model Type: SVM\n{}".format(svm_grid.best_estimator_.get_params()))
        print("Accuracy Score: {}%".format(round(svm_grid.best_score_, 4)))

        print("\n\nStarting grid search using RandomForestClassifier!")

        # run grid search
        rf_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params["rf"])
        rf_grid.fit(
            self.marvin_dataset['X_train'],
            self.marvin_dataset['y_train']
        )

        print("Model Type: RF\n{}".format(rf_grid.best_estimator_.get_params()))
        print("Accuracy Score: {}%".format(round(rf_grid.best_score_, 4)))

        self.marvin_model = {
            'svm': svm_grid,
            'rf': rf_grid
        }

