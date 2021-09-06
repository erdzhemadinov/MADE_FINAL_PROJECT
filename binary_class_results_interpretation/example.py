#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:25:11 2020

@author: anton
"""

import pandas as pd
import catboost

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             )
from explainResultsToHTML import ExplainResultsToHTML
# =============================================================================
# Data preparation
# =============================================================================
breast_cancer = load_breast_cancer()
breast_cancer_df = pd.DataFrame(breast_cancer.data,
                                columns=breast_cancer.feature_names,
                                )
breast_cancer_df['target'] = pd.Series(breast_cancer.target)

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_df.drop('target', axis=1).values,
                                                    breast_cancer_df['target'].values,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42,
                                                    )
# =============================================================================
# CatBoost classifier create and fit
# =============================================================================
ctb = catboost.CatBoostClassifier(iterations=100,
                                  learning_rate=0.1,
                                  random_seed=42,
                                  depth=5,
                                  task_type="CPU",
                                  custom_loss='AUC',
                                  l2_leaf_reg=5,
                                  use_best_model=True,
                                 )

ctb.fit(X_train,
        y_train,
        eval_set=(X_test, y_test),
        #verbose=True,
        verbose_eval=True,
        plot=False,
        early_stopping_rounds=10,
       )
# =============================================================================
# Predict targets and proba
# =============================================================================
y_train_pred_ctb = ctb.predict(X_train)
y_test_pred_ctb = ctb.predict(X_test)
y_train_pred_scores = ctb.predict_proba(X_train)[:, 1]
y_test_pred_scores = ctb.predict_proba(X_test)[:, 1]
# =============================================================================
# Classifier qualite check
# =============================================================================
print("ROC-AUC metric: train: {}, test: {}".format(roc_auc_score(y_train,
                                                                 y_train_pred_scores),
                                                   roc_auc_score(y_test,
                                                                 y_test_pred_scores),
                                                   ),
      )
print("Accuracy metric: train: {}, test: {}".format(accuracy_score(y_train,
                                                                   y_train_pred_ctb),
                                                    accuracy_score(y_test,
                                                                   y_test_pred_ctb),
                                                    ),
      )
print("F1-score metric: train: {}, test: {}".format(f1_score(y_train,
                                                             y_train_pred_ctb),
                                                    f1_score(y_test,
                                                             y_test_pred_ctb),
                                                    ),
      )
print("Precision metric: train: {}, test: {}".format(precision_score(y_train,
                                                                     y_train_pred_ctb),
                                                     precision_score(y_test,
                                                                     y_test_pred_ctb),
                                                     ),
      )
print("Recall metric: train: {}, test: {}".format(recall_score(y_train,
                                                               y_train_pred_ctb),
                                                  recall_score(y_test,
                                                               y_test_pred_ctb),
                                                  ),
      )
# =============================================================================
# Plot individual SHAP value for observation and save as .html
# =============================================================================
expl = ExplainResultsToHTML(model=ctb,
                      X_train=X_train,
                      model_type='tree_based',
                      is_proba=True, 
                     )
id = 20

expl.single_plot(breast_cancer_df.drop('target', axis=1).columns,
                 X_test[id, :],
                 )
print('\nPredicted class: {}, proba: {}'.format(ctb.predict(X_test[id, :]),
                                              ctb.predict_proba(X_test[id, :]),
                                              ),
      )
# =============================================================================
# Summary plot features importance and save as .jpeg
# =============================================================================
expl.summary_plot(breast_cancer_df.drop('target', axis=1).columns,
                  X_test,
                  is_bar=True,
                 )