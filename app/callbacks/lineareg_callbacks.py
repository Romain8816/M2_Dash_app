# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:56:08 2021
@author: Inès
"""
# Importation
import dash
from tkinter.constants import NONE
from dash import dcc
from dash import html
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Collapse import Collapse
from dash_bootstrap_components._components.Row import Row
from numpy.core.numeric import cross
from numpy.random.mtrand import random_integers
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output,State
import os
import pandas as pd
import json
from dash.exceptions import PreventUpdate
from dash import dash_table
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import validation_curve, GridSearchCV

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from scipy import stats

from fonctions.various_functions import allowed_files, get_pandas_dataframe, parse_contents
from layout.layout import location_folder, dataset_selection, target_selection,features_selection
from layout.layout import regression_tabs, classification_tabs
from fonctions.various_functions import allowed_files, get_pandas_dataframe, parse_contents
from fonctions.algo_functions import *
from fonctions.various_functions import *



#                                                       #
#           Callback - Régression linéaire
#                                                       #

def Gridsearch(app):
    #GridSearch Ok
    @app.callback(
        Output(component_id='res_Linear_GridSearchCV',component_property='children'),
        Output(component_id="ls-loading-output-0_linear", component_property="children"),
        State(component_id='lr_test_size',component_property='value'),
        State(component_id='lr_random_state',component_property='value'),
        State(component_id='lr_centrer_reduire',component_property='value'),
        Input(component_id='Linear_button_GridSearchCV',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='num_variables',component_property='data'),
        State(component_id='Linear_GridSearchCV_number_of_folds',component_property='value'),
        State(component_id='Linear_GridSearchCV_scoring',component_property='value'),
        State(component_id='Linear_GridSearchCV_njobs',component_property='value'))
    def GridSearch_linear(test_size,random_state,centrer_reduire,n_clicks,file,target,features,num_variables,nb_folds,score,nb_njobs):
        if (n_clicks == 0):
            return "",""
        else:
            if nb_njobs == "None":
                nb_njobs = None

            if score == "RMSE" or score == "MSE":
                scoring = 'neg_mean_squared_error'
            else :
                scoring = 'neg_mean_absolute_error'
            df = get_pandas_dataframe(file)
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)
            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=y,random_state=random_state,test_size=test_size)

            # défini certain paramètre à utilisé
            params = {"fit_intercept":[True,False],"copy_X":[True,False],
                      "n_jobs":[None,1,2,5,10],"positive":[True,False]}
            grid_search = get_best_params(X_train, y_train, "Regression lineaire", params, cv=nb_folds, scoring=scoring,njobs=nb_njobs)

            if score == "RMSE":
                sc = np.sqrt(abs(grid_search[0].best_score_))
                score == "RMSE"
            else :
                sc = abs(grid_search[0].best_score_)

            return html.Div(
                ["GridSearchCV best parameters : {}".format(grid_search[0].best_params_),
                 html.Br(),html.Br(),"GridSearchCV best",
                 html.B(" {} ".format(score)),": ",
                 html.B(["{:.2f}".format(sc)],
                        style={'color': 'blue'}),html.Br(),
                 html.Br(),"time : {:.2f} sec".format(grid_search[1])]),""


def FitPredict(app):
    # fit -predit (Ok juste peut etre problème de metric)
    @app.callback(
        Output(component_id='res_Linear_FitPredict', component_property='children'),
        Output(component_id='ls-loading-output-1_Linear', component_property='children'),
        State(component_id='lr_test_size',component_property='value'),
        State(component_id='lr_random_state',component_property='value'),
        State(component_id='lr_centrer_reduire',component_property='value'),
        Input('Linear_button_FitPredict','n_clicks'),
        [
        State('model_selection','value'),
        State('target_selection','value'),
        State('features_selection','value'),
        State('num_variables','data'),
        State('file_selection','value'),
        State('fit_intercept','value'),
        State('copy_X','value'),
        State('n_jobs','value')])
    def fit_predict_functionlinear(test_size,random_state,centrer_reduire,n_clicks,model,target,features,num_variables,file,fit_intercept,copy_X,n_jobs):
        #creation du dataframe

        if n_clicks == 0:
            #print(n_clicks)
            raise PreventUpdate
        else :
            t1 = time.time()
            #print(n_clicks)
            df = get_pandas_dataframe(file)
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)


                # prendre en compte le parametre None
            if fit_intercept == 'True':
                fit_intercept = True
            else :
                fit_intercept = False

            if copy_X == 'True':
                copy_X = True
            else :
                copy_X = False

            # separt en test et apprentissage
            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=y,random_state=random_state,test_size=test_size)


            #creation du model
            LinearReg = buid_linearReg(fit_intercept, copy_X, n_jobs)
            LinearReg.fit(X_train,y_train)
            #prediction

            y_pred = LinearReg.predict(X_test)
            #affichage graphique des prédictions réalisé
            t2 = time.time()

            #calcul des coeficient directeur de la droite
            def predict(x):
                return a * x + b

            a, b, r_value, p_value, std_err = stats.linregress(X_test.iloc[:,0],y_pred)
            fitline = predict(X_test.iloc[:,0])

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=X_test.iloc[:,0],
                y=y_pred,
                mode='markers',
                name='y_pred',
                marker={'size': 8, "opacity":0.8}
            ))

            fig.add_trace(go.Scatter(
                x=X_test.iloc[:,0],
                y=y_test,
                mode='markers',
                name='y_test',
                marker={'size': 8, "opacity":0.5}
            ))
            fig.add_trace(go.Scatter(
                x=X_test.iloc[:,0],
                y=fitline,
                name='regression'))

            fig.update_layout(
                title="Comparaison des points prédits avec les points tests",
                xaxis_title="X",
                yaxis_title="Y",
                legend_title="",
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                    color="black"
                )
            )
            diff = t2-t1
            return html.Div([
                html.B("Carré moyen des erreurs (MSE) "),": {:.2f}".format(mean_squared_error(y_test, y_pred)),html.Br(),html.Br(),
                html.B("Erreur quadratique moyenne (RMSE) "),": {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))),html.Br(),html.Br(),
                html.B("Coéfficient de détermination (R2) "),": {:.2f}".format(r2_score(y_test, y_pred)),html.Br(),html.Br(),
                "temps : {:.2f} sec".format(diff),html.Br(),html.Br(),
                dcc.Graph(id='res_Linear_FitPredict_graph', figure=fig),html.Br(),html.Br(),
                #dcc.Graph(id='res_regLinear_FitPredict_graph', figure=fig2),html.Br(),html.Br(),
                             ]),""


def CrossValidation(app) :
    # Cross Validation (Ok )
    @app.callback(
        Output(component_id='res_Linear_CrossValidation',component_property='children'),
        Output(component_id="ls-loading-output-2_Linear", component_property="children"),
        State(component_id='lr_centrer_reduire',component_property='value'),
        Input(component_id='Linear_button_CrossValidation',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State('num_variables','data'),
        State('fit_intercept','value'),
        State('copy_X','value'),
        State('n_jobs','value'),
        State(component_id='Linear_cv_number_of_folds',component_property='value'),
        State(component_id='Linear_cv_scoring',component_property='value'))
    def CV_score_linear(centrer_reduire,n_clicks,file,target,features,num_variables,fit_intercept,copy_X,n_jobs,cv_number_of_folds,cv_scoring):
        if (n_clicks == 0):
            return "",""
        else:

            if fit_intercept == 'True':
                fit_intercept = True
            else :
                fit_intercept = False

            if copy_X == 'True':
                copy_X = True
            else :
                copy_X = False
            if cv_scoring == "RMSE" or cv_scoring == "MSE":
                scoring = 'neg_mean_squared_error'
            else :
                scoring = 'neg_mean_absolute_error'

            df = get_pandas_dataframe(file)

            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,Y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)

            LinearReg = buid_linearReg(fit_intercept, copy_X, n_jobs)

            res = cross_validation(clf=LinearReg,X=X,Y=Y,cv=cv_number_of_folds,scoring=scoring)
            if cv_scoring == "RMSE":
                metric = np.sqrt(abs(np.mean(res[0])))
            else :
                metric = abs(np.mean(res[0]))

            return html.Div([
                "cross validation ",html.B("{} : ".format(cv_scoring)),
                html.B(["{:.2f}".format(metric)],style={'color': 'green'}),html.Br(),
                html.Br(),"time : {:.2f} sec".format(res[1])]),""
