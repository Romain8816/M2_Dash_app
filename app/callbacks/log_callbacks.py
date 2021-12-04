from tkinter.constants import NONE
from dash import dcc
from dash import html
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Collapse import Collapse
from dash_bootstrap_components._components.Row import Row
from numpy.core.numeric import cross
from numpy.random.mtrand import RandomState, random_integers
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
from layout.layout import location_folder, dataset_selection, target_selection,features_selection
from layout.layout import regression_tabs, classification_tabs

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score
from math import sqrt
from matplotlib import pyplot

from fonctions.various_functions import get_pandas_dataframe, binariser, centrer_reduire_norm, split_train_test, pre_process
from fonctions.algo_functions import build_KNeighborsRegressor, cross_validation, get_best_params, build_model

from sklearn.linear_model import LogisticRegression
# (Régression) Logistique

def Gridsearch(app):
    @app.callback(
        #Output(component_id='svr-ls-loading-output-1',component_property='children'),
        Output(component_id='res_log_GridSearchCV',component_property ='children'),   # Affichage des meilleurs paramètres 
        Input(component_id='log_button_GridSearchCV',component_property ='n_clicks'), # Validation du Gridsearch
        State(component_id='file_selection',component_property ='value'),
        State(component_id='target_selection',component_property ='value'),
        State(component_id='features_selection',component_property ='value'),
        State(component_id='log_test_size',component_property ='value'),
        State(component_id='log_shuffle',component_property='value'),
        State(component_id='log_random_state',component_property ='value'),
        State(component_id='log_stratify', component_property = 'value'),
        State(component_id='log_centrer_reduire',component_property='value'),
        State(component_id='log_gridCV_k_folds',component_property='value'),
        State(component_id='log_GridSearchCV_njobs',component_property='value'),
        State(component_id='log_gridCV_scoring',component_property='value'))

    def GridSearchCV_score (n_clicks, file, target, features, test_size, shuffle, random_state, stratify, centrer_reduire, k_fold, njobs, metric):

        if (n_clicks==0):
            PreventUpdate
        else:
            t1 = time.time()
            if njobs == "None":
                njobs = None

            df = get_pandas_dataframe(file)
            X= df[features]
            y= df[target]

            X_train, X_test, y_train, y_test = split_train_test(X, y, test_size = test_size, random_state = random_state, shuffle = shuffle, stratify = stratify)

            model = build_model(centrer_reduire, LogisticRegression)

            params = {
                'clf__penalty':['none','l1','l2','elasticnet'],
                'clf__l1_ratio' : [i for i in np.arange(0.1,0.9,0.1)],
                'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'clf__C' : [i for i in np.arange(0.1,2,0.2)],
            }

            
            grid = GridSearchCV(model,params,cv=k_fold,n_jobs=njobs,scoring=metric)
            grid.fit(X_train,y_train)

            best_params = pd.Series(grid.best_params_,index=grid.best_params_.keys())
            best_params = pd.DataFrame(best_params)
            best_params.reset_index(level=0, inplace=True)
            best_params.columns = ["paramètres","valeur"]

            t2 = time.time()
            diff = t2 - t1

            y_pred = grid.predict(X_test)

            return (
                [
                    "GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),
                    dash_table.DataTable(
                        columns=[{"name": i, "id": i} for i in best_params.columns],
                        data=best_params.to_dict('records'),
                        style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in best_params.columns]),
                        html.Br(),html.Br(),
                        "GridSearchCV meilleur ",html.B(" {} ".format(metric)),": ",html.B(["{:.4f}".format(abs(grid.best_score_))],style={'color': 'blue'}),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff)
                ]
            )

def FitPredict(app):
    # Fit et predict
    @app.callback(
        Output('res_log_FitPredict','children'),
        Input('log_button','n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='log_test_size',component_property='value'),
        State(component_id='log_random_state',component_property='value'),
        State(component_id='log_shuffle',component_property='value'),
        State(component_id='log_stratify', component_property = 'value'),
        State(component_id='log_centrer_reduire',component_property='value'),
        State(component_id='log_solver',component_property='value'),          # Noyau
        State(component_id='log_regularisation',component_property='value'),  # C
        State(component_id='log_penalty',component_property='value'),
        State(component_id='log_penalty',component_property='value'),
        State('log_degre','value'))

    def fit_predict_log (n_clicks,file,target,features,test_size,random_state, shuffle, stratify, centrer_reduire, solver, regularisation, penalty, multi_class):

        if (n_clicks == 0):
            PreventUpdate
        else:
            df = get_pandas_dataframe(file)

            X= df[features]
            y= df[target]

            X_train,X_test,y_train,y_test = split_train_test(X,y,test_size=test_size,random_state=random_state,stratify=stratify,shuffle=shuffle)

            params = {
                "multi_class": multi_class,
                "penalty" : penalty,
                "solver" : solver,
                "C" : regularisation
            }

            model = build_model(centrer_reduire,LogisticRegression,**params)
            
            
            rsquared = score['test_r2'].mean()
            mse = score['test_neg_mean_squared_error'].mean()

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            fig = px.imshow(df.corr())


            return [
                        html.Div(
                            [
                                dbc.Label("Validation score"),
                                html.P('R² : '+str(rsquared)),
                                html.P('MSE : '+str(mse))
                            ]
                        ),
                        dcc.Graph(figure=fig)
                    ]
