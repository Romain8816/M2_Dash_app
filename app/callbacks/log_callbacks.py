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
        State(component_id='log_solver',component_property='value'),
        State(component_id='log_regularisation',component_property='value'),
        State(component_id='log_penalty',component_property='value'),
        State(component_id='log_l1_ratio',component_property='value'))
    def fit_predict_log (n_clicks,file,target,features,test_size,random_state, shuffle, stratify, centrer_reduire, solver, regularisation, penalty, l1_ratio):

        if (n_clicks == 0):
            PreventUpdate
        else:
            t1 = time.time()
            df = get_pandas_dataframe(file)

            X= df[features]
            y= df[target]

            X_train,X_test,y_train,y_test = split_train_test(X,y,test_size=test_size,random_state=random_state,stratify=stratify,shuffle=shuffle)

            params = {
                "l1_ratio": l1_ratio,
                "penalty" : penalty,
                "solver" : solver,
                "C" : regularisation
            }

            model = build_model(centrer_reduire,LogisticRegression,**params)
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            labels = np.unique(y_test)


            # Matrice de confusion
            df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels = labels),columns=labels, index=labels) # matrice de confusion
            df_cm.insert(0, target, df_cm.index)
            t2 = time.time() # stop

            diff = t2 - t1

            if len(features) > 1:
                pca = PCA(n_components=2)
                temp = pca.fit_transform(X_test)
                coord = pd.DataFrame(temp,columns=["PCA1","PCA2"]) # calcul des coordonnées pour l'ACP
                y_pred = pd.DataFrame(y_pred,columns=["Log_reg_clusters"])
                y_test = pd.DataFrame(y_test.values,columns=[target])

                result = pd.concat([coord,y_pred,y_test],axis=1)

                fig_knn = px.scatter(result, x="PCA1", y="PCA2", color="Log_reg_clusters", hover_data=['Log_reg_clusters'],
                                 title="PCA des classes prédites par le modèle".format(file.split("/")[-1]))

                fig_input_data = px.scatter(result, x="PCA1", y="PCA2", color=target, hover_data=[target],
                                 title="PCA du jeu de données test")

                if len(set(list(y))) > 2: # si le nombre de classe de la variable explicative est > 2 (non binaire), on renvoie les métriques pertinentes
                    return html.Div(
                        [
                            "Matrice de confusion : ",html.Br(),
                            dash_table.DataTable(
                                id='log_cm',
                                columns=[{"name": i, "id": i} for i in df_cm.columns],
                                data=df_cm.to_dict('records'),
                                style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],
                            ),html.Br(),
                            html.B("f1_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(f1_score(y_test, y_pred,average="macro"),
                            f1_score(y_test, y_pred,average="micro"),
                            f1_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),
                            html.B("recall_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(recall_score(y_test, y_pred,average="macro"),
                            recall_score(y_test, y_pred,average="micro"),
                            recall_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),
                            html.B("precision_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(precision_score(y_test, y_pred,average="macro"),
                            precision_score(y_test, y_pred,average="micro"),
                            precision_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),
                            html.B("accuracy_score ")," {:.4f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),
                            "temps : {:.4f} sec".format(diff),html.Br(),dcc.Graph(id='res_log_FitPredict_knngraph', figure=fig_knn),
                            dcc.Graph(id='res_log_FitPredict_inputgraph', figure=fig_input_data)
                            ]
                        )
                else:
                    return html.Div(
                        [
                            "Matrice de confusion : ",html.Br(),
                            dash_table.DataTable(
                                id='log_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],
                                data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),
                                html.B("f1_score "),"binary {:.4f}".format(f1_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(y))))[0])),html.Br(),html.Br(),
                                html.B("recall_score "),"binary {:.4f}".format(recall_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(y))))[0])),html.Br(),html.Br(),
                                html.B("precision_score "),"binary {:.4f}".format(precision_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(y))))[0])),html.Br(),html.Br(),
                                html.B("accuracy_score "),"{:.4f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff),html.Br(),
                                dcc.Graph(id='res_log_FitPredict_knngraph', figure=fig_knn),
                                dcc.Graph(id='res_log_FitPredict_inputgraph', figure=fig_input_data)
                        ]
                    )

            if len(features) == 1:
                if len(set(list(y))) > 2: # si le nombre de classe de la variable explicative est > 2 (non binaire), on renvoie les métriques pertinentes
                    return html.Div(
                        [
                            "Matrice de confusion : ",html.Br(),
                            dash_table.DataTable(
                                id='log_cm',
                                columns=[{"name": i, "id": i} for i in df_cm.columns],
                                data=df_cm.to_dict('records'),
                                style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],
                            ),html.Br(),
                            html.B("f1_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(f1_score(y_test, y_pred,average="macro"),
                            f1_score(y_test, y_pred,average="micro"),
                            f1_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),
                            html.B("recall_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(recall_score(y_test, y_pred,average="macro"),
                            recall_score(y_test, y_pred,average="micro"),
                            recall_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),
                            html.B("precision_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(precision_score(y_test, y_pred,average="macro"),
                            precision_score(y_test, y_pred,average="micro"),
                            precision_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),
                            html.B("accuracy_score ")," {:.4f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),
                            "temps : {:.4f} sec".format(diff),html.Br()
                            ]
                        )
                else:
                    return html.Div(
                        [
                            "Matrice de confusion : ",html.Br(),
                            dash_table.DataTable(
                                id='log_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],
                                data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),
                                html.B("f1_score "),"binary {:.4f}".format(f1_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(y))))[0])),html.Br(),html.Br(),
                                html.B("recall_score "),"binary {:.4f}".format(recall_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(y))))[0])),html.Br(),html.Br(),
                                html.B("precision_score "),"binary {:.4f}".format(precision_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(y))))[0])),html.Br(),html.Br(),
                                html.B("accuracy_score "),"{:.4f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff),html.Br()
                        ]
                    )



######################################
# Callback en charge de faire la validation
# croisée du modèle de regression KNN
######################################
def CrossValidation(app):
    @app.callback(
        Output(component_id='res_log_CrossValidation',component_property='children'),
        Input(component_id='log_button_CrossValidation',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='log_solver',component_property='value'),
        State(component_id='log_regularisation',component_property='value'),
        State(component_id='log_penalty',component_property='value'),
        State(component_id='log_l1_ratio',component_property='value'),
        State(component_id='log_centrer_reduire',component_property='value'),
        State(component_id='log_cv_number_of_folds',component_property='value'),
        State(component_id='log_cv_scoring',component_property='value'))

    def CV_score(n_clicks,file,target,features,solver, regularisation, penalty, l1_ratio, centrer_reduire,cv_number_of_folds,cv_scoring):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time() # start
            df = get_pandas_dataframe(file) # récupération du jeu de données

            X = df[features]
            y = df[target]

            params = {
                "l1_ratio": l1_ratio,
                "penalty" : penalty,
                "solver" : solver,
                "C" : regularisation
            }

            model = build_model(centrer_reduire,LogisticRegression,**params)
            cv_res = cross_validation(clf=model['clf'],X=X,Y=y,cv=cv_number_of_folds,scoring=cv_scoring) # validation croisée
            t2 = time.time()# stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'validation croisée'

            if isinstance(cv_res, str) == False:
                return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.4f}".format(abs(np.mean(cv_res)))],style={'color': 'green'}),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff)])
            else:
                return html.Div(["cross validation :",html.Br(),"{}".format(cv_res)])
