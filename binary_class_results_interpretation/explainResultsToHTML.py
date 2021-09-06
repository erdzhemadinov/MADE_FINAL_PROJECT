#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:55:02 2020

@author: anton
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import operator
import matplotlib

from collections import OrderedDict

class ExplainResultsToHTML():
    '''
    Results interpretation of binary classification task by SHAP library
    
    Attributes
    ----------
    model:
        trained model on train dataset
    model_type: str
        model type like 'linear'', 'tree-based' or 'neural-net'
    is_proba: bool
        flag for return probabilities on plot or raw SHAP-values
    is_proba:
        flag for return probabilities on plot or raw SHAP-values
    scaler:
        scaler for inverse features values on plot (default None)
    
    Methods
    -------
    single_plot()
        Plot individual SHAP value for observation and save as .html
    summary_plot
        Summary plot SHAP value for features and save as .jpeg
    get_impact_of_n_max_shap_values
        Return impact of each top n_max features and other features
    pie_plot_impacts_by_classes
        Plot pie charts of impact features in two classes
    pie_plot_summary_impacts
        Plot pie charts of summary impact of features
        
    '''

    def __init__(self, model, X_train, model_type, is_proba, scaler=None):
        '''
        Parameters
        ----------
        model:
            trained model on train dataset
        model_type: str
            model type like 'linear'', 'tree-based' or 'neural-net'
        is_proba: bool
            flag for return probabilities on plot or raw SHAP-values
        scaler:
            scaler for inverse features values on plot (default None)
        '''
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.is_proba = is_proba
        self.scaler = scaler


    def __explainer(self):
        '''
        Creation explainer
        
        '''
        if self.model_type == 'linear':
            explainer = shap.LinearExplainer(self.model,
                                             self.X_train,
                                            )
        if self.model_type == 'tree_based':
            explainer = shap.TreeExplainer(self.model,
                                           self.X_train,
                                          )
        if self.model_type == 'neural_net':
            explainer = shap.DeepExplainer(self.model,
                                           self.X_train,
                                          )
        return explainer
    
    
    def __shap_values(self, X_test):
        '''
        Calculation SHAP values
                
        Parameters
        ----------
        X_test : numpy.ndarray
            one test data example
        '''
        explainer_ = self.__explainer()
        return explainer_.shap_values(X_test)
   

    def single_plot(self, features_list, one_row, path_save="single_plot.html"):
        '''
        Plot individual SHAP value for observation and save as .html
        
        Parameters
        ----------
        features_list : list
            list of features names
        one_row : numpy.ndarray
            one test data example
        '''
        link_ = 'logit' if self.is_proba else 'identity'
        features_ = one_row if self.scaler is None else self.scaler.inverse_transform(one_row)
        single_plot = shap.force_plot(base_value=self.__explainer().expected_value, 
                                      shap_values=self.__shap_values(one_row),
                                      features=features_,
                                      feature_names=features_list,
                                      show=False,
                                      matplotlib=False,
                                      link=link_,
                                     )
        shap.save_html(path_save, single_plot)
    
    
    def summary_plot(self, features_list, X_test, max_display, path_save="summary_plot.html", is_bar=False):
        '''
        Summary plot SHAP value for features and save as .jpeg
        
        Parameters
        ----------
        features_list : list
            list of features names
        X_test : numpy.ndarray
            test data
        is_bar : bool
            bar or dot flag
        '''
        plot_type = 'bar'if is_bar else 'dot'
        summary_plot = shap.summary_plot(self.__shap_values(X_test),
                                         X_test,
                                         max_display=max_display,
                                         feature_names=features_list,
                                         plot_type=plot_type,
                                         show=False,
                                        )
        plt.savefig('summary_plot.jpeg', bbox_inches='tight')


    def get_impact_of_n_max_shap_values(self, test_data, features_list, n_max, is_pos):
        '''
        Return impact of each top n_max features and other features
        
        Parameters
        ----------
        test_data : numpy.ndarray
            one test data example
        features_list : list
            list of features names
        n_max : int
            number of most important features
        is_pos : bool
            positive or negative class
        '''
        if test_data.ndim == 1:
            shap = self.__shap_values(test_data)
        else:
            shap = self.__shap_values(test_data).mean(0)
        shap_dict = dict(zip(features_list,
                             shap,
                             ),
                         )
        shap_pos_sum = np.sum(shap[np.where(shap > 0)])
        shap_neg_sum = np.sum(shap[np.where(shap < 0)])
        shap_dict_pos = {}
        shap_dict_neg = {}
        for key, value in shap_dict.items():
            if shap_dict[key] > 0:
                shap_dict_pos.update({key: value / shap_pos_sum})
            if shap_dict[key] < 0:
                shap_dict_neg.update({key: value / shap_neg_sum})
        if is_pos:
            other_sum = sum(dict(sorted(shap_dict_pos.items(), key = operator.itemgetter(1), reverse=True)[n_max:]).values())
            dict_ = dict(sorted(shap_dict_pos.items(), key = operator.itemgetter(1), reverse=True)[:n_max])
            dict_['Остальное'] = other_sum
            return dict(sorted(dict_.items(), key = operator.itemgetter(1), reverse=True))
        else:
            other_sum = sum(dict(sorted(shap_dict_neg.items(), key = operator.itemgetter(1), reverse=True)[n_max:]).values())
            dict_ = dict(sorted(shap_dict_neg.items(), key = operator.itemgetter(1), reverse=True)[:n_max])
            dict_['Остальное'] = other_sum
            return dict(sorted(dict_.items(), key = operator.itemgetter(1), reverse=True))
    
    
    def pie_plot_impacts_by_classes(self, pos_imp, neg_imp):
        '''
        Plot pie charts of impact features in two classes
        
        Parameters
        ----------
        pos_imp : dict
            dict with feature as key and value as impact 
        neg_imp : dict
            number of most important features
        '''
        figsize = (10, 10)
        #Positive pie chart plot
        labels = list(pos_imp.keys())
        sizes = list(pos_imp.values())
        other_explode_ind = list(OrderedDict(pos_imp).keys()).index('Остальное')
        explode = [0.05] * (len(labels) - 1)
        explode.insert(other_explode_ind, 0.15)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ff0d57", "#f2f2f2"])
        colors = cmap(np.linspace(0., 1., len(sizes)))
        plt.figure(figsize=figsize)
        plt.pie(sizes,
               explode=explode,
               labels=labels,
               autopct='%1.1f%%',
               shadow=True,
               colors=colors,
               startangle=90,
               pctdistance=0.8,
               )
        centre_circle = plt.Circle((0, 0),
                                    0.70,
                                    fc='white',
                                    )
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.tight_layout()
        plt.savefig('pos.png', dpi=100, bbox_inches='tight')
        plt.show()
        # Negative pie chart plot
        labels = list(neg_imp.keys())
        sizes = list(neg_imp.values())
        other_explode_ind = list(OrderedDict(neg_imp).keys()).index('Остальное')
        explode = [0.05] * (len(labels) - 1)
        explode.insert(other_explode_ind, 0.15)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1e88e5", "#f2f2f2"])
        colors = cmap(np.linspace(0., 1., len(sizes)))
        plt.figure(figsize=figsize)
        plt.pie(sizes,
               explode=explode,
               labels=labels,
               autopct='%1.1f%%',
               shadow=True,
               colors=colors,
               startangle=90,
               pctdistance=0.8,
               )
        centre_circle = plt.Circle((0, 0),
                                    0.70,
                                    fc='white',
                                    )
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.tight_layout()
        plt.savefig('neg.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        
    def pie_plot_summary_impacts(self, test_data, features_list, n_max=5):
        '''
        Plot pie charts of summary impact of features
        
        Parameters
        ----------
        test_data : numpy.ndarray
            test data
        features_list : dict
             list of features names
        n_max : int
            number of most important features
        '''
        figsize = (10, 10)
        vals = np.abs( self.__shap_values(test_data)).mean(0)
        feature_importance = dict(zip(features_list, vals))
        feature_importance_ = {}
        for key, value in feature_importance.items():
            feature_importance_.update({key: value / np.sum(vals)})
        p_list = sorted(feature_importance_.items(), key = operator.itemgetter(1), reverse=True)
        p_dict_n_max = dict(p_list[:n_max])
        p_dict_n_max['Остальное'] = sum(dict(p_list[n_max:]).values())
        d = dict(sorted(p_dict_n_max.items(), key=operator.itemgetter(1), reverse=True))
        labels = list(d.keys())
        sizes = list(d.values())
        other_explode_ind = list(OrderedDict(d).keys()).index('Остальное')
        explode = [0.05] * (len(labels) - 1)
        explode.insert(other_explode_ind, 0.15)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#4f9b17", "#f2f2f2"])
        colors = cmap(np.linspace(0., 1., len(sizes)))
        plt.figure(figsize=figsize)
        plt.pie(sizes,
               explode=explode,
               labels=labels,
               autopct='%1.1f%%',
               shadow=True,
               colors=colors,
               startangle=90,
               pctdistance=0.8,
               )
        centre_circle = plt.Circle((0, 0),
                                    0.70,
                                    fc='white',
                                    )
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.tight_layout()
        plt.savefig('pie_all_impact.png', dpi=100, bbox_inches='tight')
        plt.show()
        