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
from sklearn.svm import SVR

from layout.layout import location_folder, dataset_selection, target_selection,features_selection
from layout.layout import regression_tabs, classification_tabs

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score, mean_absolute_error
from math import sqrt
from matplotlib import pyplot

from fonctions.various_functions import get_pandas_dataframe, binariser, centrer_reduire_norm, split_train_test, pre_process
from fonctions.algo_functions import build_smv, build_KNeighborsRegressor, cross_validation, get_best_params, build_model, cross_validation

# (Régression) svr

def Gridsearch(app):
    @app.callback(
        #Output(component_id='svr-ls-loading-output-1',component_property='children'),
        Output(component_id='res_svr_GridSearchCV',component_property='children'),   # Affichage des meilleurs paramètres 
        Input(component_id='svr_button_GridSearchCV',component_property='n_clicks'), # Validation du Gridsearch
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='svr_train_size',component_property='value'),
        State(component_id='svr_random_state',component_property='value'),
        State(component_id='svr_centrer_reduire',component_property='value'),
        State(component_id='svr_gridCV_k_folds',component_property='value'),
        State(component_id='svr_GridSearchCV_njobs',component_property='value'),
        State(component_id='svr_gridCV_scoring',component_property='value'))

    def GridSearchCV_score (n_clicks, file, target, features, train_size, random_state, centrer_reduire, k_fold, njobs, metric):
        if (n_clicks==0):
            PreventUpdate
        else:
            t1 = time.time()
            if njobs == "None":
                njobs = None

            df = get_pandas_dataframe(file)
            X= df[features]
            y= df[target]

            X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_size,random_state = random_state)

            model = build_model(centrer_reduire,SVR)

            params = {
                'clf__kernel':['linear','poly','rbf','sigmoid'],
                'clf__degree': [i for i in range(1,4)],
                'clf__gamma': ['scale','auto'],
                'clf__coef0': [i for i in np.arange(0.1,1,0.3)],
                'clf__C' : [i for i in np.arange(0.1,1,0.3)],
                'clf__epsilon' : [i for i in np.arange(0.1,1,0.3)]
            }
            
            if (metric=="RMSE"):
                grid = GridSearchCV(model,params,scoring="neg_mean_squared_error",cv=k_fold,n_jobs=njobs)
                grid.fit(X_train,y_train)
                grid.best_score_ = np.sqrt(abs(grid.best_score_))
            else:
                grid = GridSearchCV(model,params,scoring=metric,cv=k_fold,n_jobs=njobs)
                grid.fit(X_train,y_train)
                grid.best_score_ = abs(grid.best_score_)

            best_params = pd.Series(grid.best_params_,index=grid.best_params_.keys())
            best_params = pd.DataFrame(best_params)
            best_params.reset_index(level=0, inplace=True)
            best_params.columns = ["paramètres","valeur"]

            if (metric=="RMSE"):
                grid = GridSearchCV(model,params,scoring="neg_mean_squared_error",cv=k_fold,n_jobs=njobs)
                grid.fit(X_train,y_train)
                grid.best_score_ = np.sqrt(abs(grid.best_score_))
            else:
                grid = GridSearchCV(model,params,scoring=metric,cv=k_fold,n_jobs=njobs)
                grid.fit(X_train,y_train)
                grid.best_score_ = abs(grid.best_score_)
                
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
        Output('res_svr_FitPredict','children'),
        Input('smv_button','n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='svr_test_size',component_property='value'),
        State(component_id='svr_random_state',component_property='value'),
        State(component_id='svr_centrer_reduire',component_property='value'),
        State(component_id='svr_kernel_selection',component_property='value'),          # Noyau
        State(component_id='svr_regularisation_selection',component_property='value'),  # C
        State(component_id='svr_epsilon',component_property='value'),
        State('svr_degre','value'))

    def fit_predict_svr (n_clicks,file,target,features,test_size,random_state,centrer_reduire,kernel,regularisation,epsilon,degre):
        if (n_clicks == 0):
            PreventUpdate
        else:
            t1 = time.time() # start
            df = get_pandas_dataframe(file) # récupération du jeu de données

            X= df[features]
            y= df[target]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

            params = {
                "kernel" : kernel,
                "C" : regularisation,
                "epsilon" : epsilon,
                "degree" : degre
            }

            model = build_model(centrer_reduire,SVR,**params)
            model.fit(X_train,y_train) # apprentissage
            y_pred = model.predict(X_test) # prédiction

            t2 = time.time() # stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'performance du modèle sur le jeu test'

            k = 0
            more_uniq_col = ""
            for col in X_test: # récupérer la variable explicative avec le plus de valeurs uniques pour la représentation graphique
                if len(X_test[col].unique()) > k:
                    more_uniq_col = col
                    k = len(X_test[col].unique())
            X_test = X_test.sort_values(by=more_uniq_col)

            print(more_uniq_col)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=X_train[more_uniq_col],y=y_train,mode='markers',name='train',marker={'size': 8, "opacity":0.8})
            )

            fig.add_trace(
                go.Scatter(x=X_test[more_uniq_col],y=y_test,mode='markers',name='test',marker={'size': 8, "opacity":0.5})
            )
            
            fig.add_trace(
                go.Scatter(x=X_test[more_uniq_col],y=model.fit(pd.DataFrame(X_train[more_uniq_col]),y_train).predict(pd.DataFrame(X_test[more_uniq_col])),mode='lines',name='prediction',marker={'size': 8, "opacity":0.5})
            )
            fig.update_layout(
                title="Comparaison des points prédits avec les points tests",
                xaxis_title="{}".format(more_uniq_col),
                yaxis_title="{}".format(target),
                legend_title="",
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                    color="black"
            ))

            return html.Div(
                [
                    #dbc.Label("Validation score"),
                    html.B("Carré moyen des erreurs (MSE) "),": {:.4f}".format(abs(mean_squared_error(y_test, y_pred))),html.Br(),html.Br(),
                    html.B("Erreur quadratique moyenne (RMSE) "),": {:.4f}".format(np.sqrt(abs(mean_squared_error(y_test, y_pred)))),html.Br(),html.Br(),
                    html.B("Erreur moyenne absolue (MAE) "),": {:.4f}".format(abs(mean_absolute_error(y_test, y_pred))),html.Br(),html.Br(),
                    html.B("Coefficient de détermination (R2) "),": {:.4f}".format(abs(r2_score(y_test, y_pred))),html.Br(),html.Br(),
                    "temps : {:.4f} sec".format(diff),html.Br(),html.Br(),
                    dcc.Graph(id='res_KNeighborsRegressor_FitPredict_knngraph', figure=fig),html.Br(),html.Br(),
                ]
            )     

def CrossValidation(app):
    @app.callback(
        Output(component_id='res_svr_CrossValidation',component_property='children'),
        Input(component_id='svr_button_CrossValidation',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='svr_centrer_reduire',component_property='value'),
        State(component_id='svr_cv_number_of_folds',component_property='value'),
        State(component_id='svr_cv_scoring',component_property='value'),
        State(component_id='svr_kernel_selection',component_property='value'),          # Noyau
        State(component_id='svr_regularisation_selection',component_property='value'),  # C
        State(component_id='svr_epsilon',component_property='value'),
        State(component_id='svr_degre',component_property = 'value'))
        
    def CV_score(n_clicks,file,target,features,centrer_reduire,k_fold,cv_scoring, kernel,regularisation,epsilon,degre):
        if (n_clicks == 0):
            PreventUpdate
        else:
            t1 = time.time() # start
            df = get_pandas_dataframe(file)

            X = df[features]
            y = df[target]

            params = {
                "kernel" : kernel,
                "C" : regularisation,
                "epsilon" : epsilon,
                "degree" : degre
            }

            model = build_model(centrer_reduire,SVR,**params)

            if cv_scoring == "MAE":
                cv_res = cross_val_score(estimator=model, X=X, y=y, cv = k_fold, scoring = "neg_mean_absolute_error")
                cv_res = abs(np.mean(cv_res))
            elif cv_scoring == "RMSE" :
                cv_res = cross_val_score(estimator=model, X=X, y=y, cv = k_fold, scoring = "neg_mean_squared_error")
                cv_res = np.sqrt(abs(np.mean(cv_res)))
            else:
                cv_res = cross_val_score(estimator=model, X=X, y=y, cv = k_fold, scoring = "neg_mean_squared_error")
                cv_res = abs(np.mean(cv_res))



            t2 = time.time()# stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'validation croisée'
            return html.Div(
                [
                    "cross validation ",html.B("{} : ".format(cv_scoring)),
                    html.B(["{:.4f}".format(cv_res)],style={'color': 'green'}),html.Br(),html.Br(),
                    "temps : {:.4f} sec".format(diff)
                ]
            )