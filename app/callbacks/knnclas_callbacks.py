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

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score
from math import sqrt
from matplotlib import pyplot

# GridSearchCV
def Gridsearch(app):
    @app.callback(
        Output(component_id='res_KNeighborsClassifier_GridSearchCV',component_property='children'),
        Output(component_id="KNeighborsClassifier-ls-loading-output-1", component_property="children"),
        Input(component_id='KNeighborsClassifier_button_GridSearchCV',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='num_variables',component_property='data'),
        State(component_id='KNeighborsClassifier_centrer_reduire',component_property='value'),
        State(component_id='KNeighborsClassifier_GridSearchCV_number_of_folds',component_property='value'),
        State(component_id='KNeighborsClassifier_GridSearchCV_scoring',component_property='value'),
        State(component_id='KNeighborsClassifier_GridSearchCV_njobs',component_property='value'))
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
            grid_search = get_best_params(X=X,Y=Y,clf="KNeighborsClassifier",params=params,cv=GridSearchCV_number_of_folds,scoring=GridSearchCV_scoring,njobs=njobs)
            t2 = time.time()

            params_opti = pd.Series(grid_search.best_params_,index=grid_search.best_params_.keys())
            params_opti = pd.DataFrame(params_opti)
            params_opti.reset_index(level=0, inplace=True)
            params_opti.columns = ["paramètres","valeur"]

            diff = t2 - t1
            if GridSearchCV_scoring == "RMSE":
                return html.Div(["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),dash_table.DataTable(id='KNeighborsClassifier_params_opti',columns=[{"name": i, "id": i} for i in params_opti.columns],data=params_opti.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in params_opti.columns]),html.Br(),html.Br(),"GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{:.2f}".format(sqrt(abs(grid_search.best_score_)))],style={'color': 'blue'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""
            else:
                return html.Div(["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),dash_table.DataTable(id='KNeighborsClassifier_params_opti',columns=[{"name": i, "id": i} for i in params_opti.columns],data=params_opti.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in params_opti.columns]),html.Br(),html.Br(),"GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{:.2f}".format(abs(grid_search.best_score_))],style={'color': 'blue'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""

# FitPredict
def FitPredict(app):
    @app.callback(
        Output(component_id='res_KNeighborsClassifier_FitPredict',component_property='children'),
        Output(component_id="KNeighborsClassifier-ls-loading-output-3", component_property="children"),
        Input(component_id='KNeighborsClassifier_button_FitPredict',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='num_variables',component_property='data'),
        State(component_id='KNeighborsClassifier_n_neighbors',component_property='value'),
        State(component_id='KNeighborsClassifier_weights',component_property='value'),
        State(component_id='KNeighborsClassifier_algorithm',component_property='value'),
        State(component_id='KNeighborsClassifier_leaf_size',component_property='value'),
        State(component_id='KNeighborsClassifier_p',component_property='value'),
        State(component_id='KNeighborsClassifier_metric',component_property='value'),
        State(component_id='KNeighborsClassifier_centrer_reduire',component_property='value'),
        State(component_id='KNeighborsClassifier_test_size',component_property='value'),
        State(component_id='KNeighborsClassifier_shuffle',component_property='value'),
        State(component_id='KNeighborsClassifier_stratify',component_property='value'))
    def CV_score(n_clicks,file,target,features,num_variables,n_neighbors,weights,algorithm,leaf_size,p,metric,centrer_reduire,test_size,shuffle,stratify):
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
            clf = build_KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
            if shuffle == "True":
                shuffle = True
                if stratify == "False":
                    stratify = None
                if stratify == "True":
                    stratify = Y
            if shuffle == "False":
                shuffle = False
                stratify = None
            X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=float(test_size),shuffle=shuffle,stratify=stratify)
            clf.fit(X_train.values,y_train.values)
            y_pred = clf.predict(X_test.values)
            labels = np.unique(y_test)
            df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred,labels=labels),columns=labels, index=labels)
            df_cm.insert(0, target, df_cm.index)
            pca = PCA(n_components=2)
            temp = pca.fit_transform(X_test)
            coord = pd.DataFrame(temp,columns=["PCA1","PCA2"])
            Y_pred = pd.DataFrame(y_pred,columns=["knn_clusters"])
            Y_test = pd.DataFrame(y_test.values,columns=[target])
            result = pd.concat([coord,Y_pred,Y_test],axis=1)
            fig_knn = px.scatter(result, x="PCA1", y="PCA2", color="knn_clusters", hover_data=['knn_clusters'],
                             title="PCA du jeu de données {}, y_pred KNeighborsClassifier".format(file.split("/")[-1]))
            fig_input_data = px.scatter(result, x="PCA1", y="PCA2", color=target, hover_data=[target],
                             title="PCA du jeu de données {}, y_test".format(file.split("/")[-1]))
            t2 = time.time()
            diff = t2 - t1

            #print(pd.concat([,y_pred],axis=1))
            #fig_y_pred = px.scatter(x=, y=,color_discrete_sequence=['blue'],opacity=0.5)
            #fig_y_test = px.scatter(x=X_test.iloc[:,0], y=y_test,color_discrete_sequence=['red'],opacity=0.5)
            #fig_all = go.Figure(data=fig_y_pred.data + fig_y_test.data, name="Name of Trace 2")


            if len(set(list(Y))) > 2:
                return html.Div(["Matrice de confusion : ",html.Br(),dash_table.DataTable(id='KNeighborsClassifier_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),html.B("f1_score "),"macro {:.2f} , micro {:.2f}, weighted {:.2f}".format(f1_score(y_test, y_pred,average="macro"),f1_score(y_test, y_pred,average="micro"),f1_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("recall_score "),"macro {:.2f} , micro {:.2f}, weighted {:.2f}".format(recall_score(y_test, y_pred,average="macro"),recall_score(y_test, y_pred,average="micro"),recall_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("precision_score "),"macro {:.2f} , micro {:.2f}, weighted {:.2f}".format(precision_score(y_test, y_pred,average="macro"),precision_score(y_test, y_pred,average="micro"),precision_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("accuracy_score ")," {:.2f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff),html.Br(),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_inputgraph', figure=fig_input_data),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_knngraph', figure=fig_knn)]),""
            else:
                return html.Div(["Matrice de confusion : ",html.Br(),dash_table.DataTable(id='KNeighborsClassifier_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),html.B("f1_score "),"binary {:.2f}".format(f1_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("recall_score "),"binary {:.2f}".format(recall_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("precision_score "),"binary {:.2f}".format(precision_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("accuracy_score "),"{:.2f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff),html.Br(),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_inputgraph', figure=fig_input_data),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_knngraph', figure=fig_knn)]),""


# CrossValidation
def CrossValidation(app):
    @app.callback(
        Output(component_id='res_KNeighborsClassifier_CrossValidation',component_property='children'),
        Output(component_id="KNeighborsClassifier-ls-loading-output-2", component_property="children"),
        Input(component_id='KNeighborsClassifier_button_CrossValidation',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='num_variables',component_property='data'),
        State(component_id='KNeighborsClassifier_n_neighbors',component_property='value'),
        State(component_id='KNeighborsClassifier_weights',component_property='value'),
        State(component_id='KNeighborsClassifier_algorithm',component_property='value'),
        State(component_id='KNeighborsClassifier_leaf_size',component_property='value'),
        State(component_id='KNeighborsClassifier_p',component_property='value'),
        State(component_id='KNeighborsClassifier_metric',component_property='value'),
        State(component_id='KNeighborsClassifier_centrer_reduire',component_property='value'),
        State(component_id='KNeighborsClassifier_cv_number_of_folds',component_property='value'),
        State(component_id='KNeighborsClassifier_cv_scoring',component_property='value'))
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
            clf = build_KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
            res = cross_validation(clf=clf,X=X,Y=Y,cv=cv_number_of_folds,scoring=cv_scoring)
            t2 = time.time()
            diff = t2 - t1
            if cv_scoring == "RMSE":
                return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.2f}".format(sqrt(abs(np.mean(res))))],style={'color': 'green'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""
            else:
                return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.2f}".format(abs(np.mean(res)))],style={'color': 'green'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""
