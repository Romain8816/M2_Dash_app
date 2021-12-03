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
from fonctions.algo_functions import build_smv, build_KNeighborsRegressor, cross_validation, get_best_params

from sklearn.linear_model import LogisticRegression
# (Régression) Logistique

def Gridsearch(app):
    @app.callback(
        #Output(component_id='svr-ls-loading-output-1',component_property='children'),
        Output(component_id='res_log_GridSearchCV',component_property='children'),   # Affichage des meilleurs paramètres 
        Input(component_id='log_button_GridSearchCV',component_property='n_clicks'), # Validation du Gridsearch
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='log_train_size',component_property='value'),
        State(component_id='log_random_state',component_property='value'),
        State(component_id='log_centrer_reduire',component_property='value'),
        State(component_id='log_gridCV_k_folds',component_property='value'),
        State(component_id='log_GridSearchCV_njobs',component_property='value'),
        State(component_id='log_gridCV_scoring',component_property='value'))

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

            numerical_features = make_column_selector(dtype_include=np.number)
            categorical_features = make_column_selector(dtype_exclude=np.number)

            categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(drop='first',sparse=False))
            
            if (centrer_reduire == ['yes']):
                numerical_pipeline = make_pipeline(SimpleImputer(),StandardScaler())
            else :
                numerical_pipeline = make_pipeline(SimpleImputer())

            preprocessor = make_column_transformer((numerical_pipeline,numerical_features),
                                                (categorical_pipeline,categorical_features))


            clf = LogisticRegression(max_iter=100)

            model = Pipeline([('preprocessor',preprocessor),('clf',clf)])

            params = {
                'clf__multi_class': ['auto', 'ovr', 'multinomial'],
                'clf__penalty':['none','l1','l2','elasticnet'],
                'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'clf__C' : [i for i in np.arange(0.1,2,0.2)],
            }

            #print(clf.get_params().keys())
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
        Output('res_log_GridSearchCV','children'),
        Input('log_button','n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='log_test_size',component_property='value'),
        State(component_id='log_random_state',component_property='value'),
        State(component_id='log_k_fold',component_property='value'),
        State(component_id='log_kernel_selection',component_property='value'),          # Noyau
        State(component_id='log_regularisation_selection',component_property='value'),  # C
        State(component_id='log_epsilon',component_property='value'),
        State('log_degre','value'))

    def log (n_clicks,file,target,features,test_size,random_state,k_fold,kernel,regularisation,epsilon,degre):

        if (n_clicks == 0):
            PreventUpdate
        else:
            df = get_pandas_dataframe(file)

            X= df[features]
            y= df[target]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

            model = build_smv(kernel,regularisation,epsilon)
            score = cross_validate(model,X_train,y_train,cv=k_fold,scoring=('r2','neg_mean_squared_error'),return_train_score=True)

            rsquared = score['test_r2'].mean()
            mse = score['test_neg_mean_squared_error'].mean()

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            fig = px.imshow(df.corr())

            #fig = px.scatter_matrix(df,dimensions=features)

            # train_score, val_score = validation_curve(model,X_train,y_train,param_name='svr__C',param_range=np.arange(0,100),cv=k_fold)

            # fig = go.Figure()

            # fig.add_trace(go.Scatter(x=np.arange(0,100), y=val_score.mean(axis=1),mode='lines',name='validation score'))
            # fig.add_trace(go.Scatter(x=np.arange(0,100), y=train_score.mean(axis=1),mode='lines',name='training score'))
            # fig.update_layout(title="Score en fonction de C")

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
