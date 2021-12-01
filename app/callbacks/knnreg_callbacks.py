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

from fonctions.various_functions import get_pandas_dataframe
from fonctions.algo_functions import *
from sklearn.linear_model import LogisticRegression

from statistics import mean

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score, mean_absolute_error
from math import sqrt
from matplotlib import pyplot


# GridSearchCV
def Gridsearch(app):
    @app.callback(
        Output(component_id='res_KNeighborsRegressor_GridSearchCV',component_property='children'),
        Output(component_id="KNeighborsRegressor-ls-loading-output-1", component_property="children"),
        Input(component_id='KNeighborsRegressor_button_GridSearchCV',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='num_variables',component_property='data'),
        State(component_id='KNeighborsRegressor_centrer_reduire',component_property='value'),
        State(component_id='KNeighborsRegressor_GridSearchCV_number_of_folds',component_property='value'),
        State(component_id='KNeighborsRegressor_GridSearchCV_scoring',component_property='value'),
        State(component_id='KNeighborsRegressor_GridSearchCV_njobs',component_property='value'))
    def GridSearchCV_score(n_clicks,file,target,features,num_variables,centrer_reduire,GridSearchCV_number_of_folds,GridSearchCV_scoring,njobs):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time()
            if njobs == "None":
                njobs = None
            df = get_pandas_dataframe(file)
            check_type_heterogeneity = all(element in num_variables for element in features)
            if check_type_heterogeneity == False:
                bin = binariser(df=df,features=features,target=target)
                df = bin[0]
                features = bin[1]
            if centrer_reduire == ['yes']:
                X = centrer_reduire_norm(df=df,features=features)
            else:
                X = df[features]
            Y= df[target]
            params = {'n_neighbors':list(range(1,21)), 'weights':["uniform","distance"], 'algorithm':["auto","brute"], 'leaf_size':[5,10,20,30,40], 'p':[1,2], 'metric':["minkowski","euclidean","manhattan"]}
            grid_search = get_best_params(X=X,Y=Y,clf="KNeighborsRegressor",params=params,cv=GridSearchCV_number_of_folds,scoring=GridSearchCV_scoring,njobs=njobs)
            t2 = time.time()

            params_opti = pd.Series(grid_search.best_params_,index=grid_search.best_params_.keys())
            params_opti = pd.DataFrame(params_opti)
            params_opti.reset_index(level=0, inplace=True)
            params_opti.columns = ["paramètres","valeur"]

            diff = t2 - t1
            if isinstance(grid_search,str):
                return html.Div(["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),dash_table.DataTable(id='KNeighborsClassifier_params_opti',columns=[{"name": i, "id": i} for i in params_opti.columns],data=params_opti.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in params_opti.columns]),html.Br(),html.Br(),"GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{}".format(grid_search)],style={'color': 'red'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""
            else:
                return html.Div(["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),dash_table.DataTable(id='KNeighborsClassifier_params_opti',columns=[{"name": i, "id": i} for i in params_opti.columns],data=params_opti.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in params_opti.columns]),html.Br(),html.Br(),"GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{:.2f}".format(grid_search.best_score_)],style={'color': 'blue'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""

# FitPredict
def FitPredict(app):
    @app.callback(
        Output(component_id='res_KNeighborsRegressor_FitPredict',component_property='children'),
        Output(component_id="KNeighborsRegressor-ls-loading-output-3", component_property="children"),
        Input(component_id='KNeighborsRegressor_button_FitPredict',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='num_variables',component_property='data'),
        State(component_id='KNeighborsRegressor_n_neighbors',component_property='value'),
        State(component_id='KNeighborsRegressor_weights',component_property='value'),
        State(component_id='KNeighborsRegressor_algorithm',component_property='value'),
        State(component_id='KNeighborsRegressor_leaf_size',component_property='value'),
        State(component_id='KNeighborsRegressor_p',component_property='value'),
        State(component_id='KNeighborsRegressor_metric',component_property='value'),
        State(component_id='KNeighborsRegressor_centrer_reduire',component_property='value'),
        State(component_id='KNeighborsRegressor_test_size',component_property='value'),
        State(component_id='KNeighborsRegressor_shuffle',component_property='value'))
    def CV_score(n_clicks,file,target,features,num_variables,n_neighbors,weights,algorithm,leaf_size,p,metric,centrer_reduire,test_size,shuffle):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time()
            df = get_pandas_dataframe(file)
            check_type_heterogeneity = all(element in num_variables for element in features)
            if check_type_heterogeneity == False:
                bin = binariser(df=df,features=features,target=target)
                df = bin[0]
                features = bin[1]
            if centrer_reduire == ['yes']:
                X = centrer_reduire_norm(df=df,features=features)
            else:
                X = df[features]
            Y= df[target]
            clf = build_KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
            if shuffle == "True":
                shuffle = True
            if shuffle == "False":
                shuffle = False
            X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=float(test_size),shuffle=shuffle)
            clf.fit(X_train.values,y_train.values)
            y_pred = clf.predict(X_test.values)
            t2 = time.time()

            diff = t2 - t1

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

            return html.Div([html.B("Carré moyen des erreurs (MSE) "),": {:.2f}".format(mean_squared_error(y_test, y_pred)),html.Br(),html.Br(),
                             html.B("Erreur quadratique moyenne (RMSE) "),": {:.2f}".format(sqrt(mean_squared_error(y_test, y_pred))),html.Br(),html.Br(),
                             html.B("Erreur moyenne absolue (MAE) "),": {:.2f}".format(sqrt(mean_absolute_error(y_test, y_pred))),html.Br(),html.Br(),
                             html.B("Coéfficient de détermination (R2) "),": {:.2f}".format(r2_score(y_test, y_pred)),html.Br(),html.Br(),
                             "temps : {:.2f} sec".format(diff),html.Br(),html.Br(),
                             dcc.Graph(id='res_KNeighborsRegressor_FitPredict_knngraph', figure=fig),html.Br(),html.Br(),
                             ]),""

# CrossValidation
def CrossValidation(app):
    @app.callback(
        Output(component_id='res_KNeighborsRegressor_CrossValidation',component_property='children'),
        Output(component_id="KNeighborsRegressor-ls-loading-output-2", component_property="children"),
        Input(component_id='KNeighborsRegressor_button_CrossValidation',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='num_variables',component_property='data'),
        State(component_id='KNeighborsRegressor_n_neighbors',component_property='value'),
        State(component_id='KNeighborsRegressor_weights',component_property='value'),
        State(component_id='KNeighborsRegressor_algorithm',component_property='value'),
        State(component_id='KNeighborsRegressor_leaf_size',component_property='value'),
        State(component_id='KNeighborsRegressor_p',component_property='value'),
        State(component_id='KNeighborsRegressor_metric',component_property='value'),
        State(component_id='KNeighborsRegressor_centrer_reduire',component_property='value'),
        State(component_id='KNeighborsRegressor_cv_number_of_folds',component_property='value'),
        State(component_id='KNeighborsRegressor_cv_scoring',component_property='value'))
    def CV_score(n_clicks,file,target,features,num_variables,n_neighbors,weights,algorithm,leaf_size,p,metric,centrer_reduire,cv_number_of_folds,cv_scoring):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time()
            df = get_pandas_dataframe(file)
            check_type_heterogeneity = all(element in num_variables for element in features)
            if check_type_heterogeneity == False:
                bin = binariser(df=df,features=features,target=target)
                df = bin[0]
                features = bin[1]
            if centrer_reduire == ['yes']:
                X = centrer_reduire_norm(df=df,features=features)
            else:
                X = df[features]
            Y= df[target]
            clf = build_KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
            res = cross_validation(clf=clf,X=X,Y=Y,cv=cv_number_of_folds,scoring=cv_scoring)
            t2 = time.time()
            diff = t2 - t1
            if isinstance(res,str):
                return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{}".format(res)],style={'color': 'red'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""
            else:
                return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.2f}".format(mean(res))],style={'color': 'green'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""
