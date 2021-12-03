# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:06:13 2021
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
from fonctions.algo_functions import *
from fonctions.various_functions import *


#                                             #
#       Callback : Decision Tree Classivieur
#                                             #

def Gridsearch(app) :
    @app.callback(
        Output(component_id='res_Tree_GridSearchCV',component_property='children'),
        Output(component_id="ls-loading-output-0_tree", component_property="children"),
        State(component_id='tree_test_size',component_property='value'),
        State(component_id='tree_random_state',component_property='value'),
        State(component_id='tree_centrer_reduire',component_property='value'),
        State(component_id='tree_shuffle',component_property='value'),
        State(component_id='tree_stratify',component_property='value'),
        Input(component_id='Tree_button_GridSearchCV',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='Tree_GridSearchCV_number_of_folds',component_property='value'),
        State(component_id='Tree_GridSearchCV_scoring',component_property='value'),
        State(component_id='Tree_GridSearchCV_njobs',component_property='value'))
    def GridSearch_tree(test_size,random_state,centrer_reduire,shuffle,stratify,n_clicks,file,target,features,nb_folds,score,nb_njobs):
        if (n_clicks == 0):
            return "",""
        else:
            if nb_njobs == "None":
                nb_njobs = None

            df = get_pandas_dataframe(file)

            X = df.loc[:,features]
            y = df.loc[:,target]

            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=y,random_state=random_state,test_size=test_size,shuffle=shuffle,stratify=stratify)

            # défini certain paramètre à utilisé
            params = {"criterion":["gini","entropy"],"splitter":["best","random"],
                      "max_depth":[1,5,10],"min_samples_split":[2,5,10,20],
                      "min_samples_leaf":[1,4,8,20],"max_features":['auto','sqrt','log2']}
            grid_search = get_best_params(X_train, y_train, "Arbre de decision", params, cv=nb_folds, scoring=score, njobs=nb_njobs)
            best_params = pd.Series(grid_search[0].best_params_,index=grid_search[0].best_params_.keys())
            best_params = pd.DataFrame(best_params)
            best_params.reset_index(level=0, inplace=True)
            best_params.columns = ["paramètres","valeurs"]
            return html.Div(
                ["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),dash_table.DataTable(id='tree_params_opti',columns=[{"name": i, "id": i} for i in best_params.columns],data=best_params.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in best_params.columns]),
                 html.Br(),html.Br(),"GridSearchCV best",
                 html.B(" {} ".format(score)),": ",
                 html.B(["{:.2f}".format(grid_search[0].best_score_)],
                        style={'color': 'blue'}),html.Br(),
                 html.Br(),"temps : {:.2f} sec".format(grid_search[1])]),""


def FitPredict(app) :
    # fit -predit (Ok + affichage de l'arbre)
    @app.callback(
        Output(component_id='res_Tree_FitPredict', component_property='children'),
        Output(component_id='ls-loading-output-1_tree', component_property='children'),
        State(component_id='tree_test_size',component_property='value'),
        State(component_id='tree_random_state',component_property='value'),
        State(component_id='tree_centrer_reduire',component_property='value'),
        State(component_id='tree_shuffle',component_property='value'),
        State(component_id='tree_stratify',component_property='value'),
        Input('Tree_button_FitPredict','n_clicks'),
        Input('tree_plot_button','n_clicks'),
        #bouton pour afficher le graphe
        [State('model_selection','value'),
        State('target_selection','value'),
        State('features_selection','value'),
        State('file_selection','value'),
        State('criterion','value'),
        State('splitter','value'),
        State('max_depth','value'),
        State('min_samples_split','value'),
        State('min_samples_leaf','value'),
        State('max_leaf_nodes','value')])
    def fit_predict_function(test_size,random_state,centrer_reduire,shuffle,stratify,n_clicks,plot_clicks,model,target,feature,file,criterion,splitter,max_depth,min_samples_split,min_samples_leaf,max_leaf_nodes):

        if n_clicks == 0:
            #print(n_clicks)
            raise PreventUpdate
        else :
            t1 = time.time()
            #print(n_clicks)
            df = get_pandas_dataframe(file)
            #print(df.head(10))
            #on le fait que si model == arbre decison

                # prendre en compte le parametre None
            if max_depth == 0:
                max_depth = None
            if max_leaf_nodes == 0:
                max_leaf_nodes = None

            # separt en test et apprentissage
            X = df.loc[:,feature]
            y = df.loc[:,target]

            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=y,random_state=random_state,test_size=test_size,shuffle=shuffle,stratify=stratify)

            #creation du model
            tree = build_tree(criterion, splitter, max_depth, min_samples_split,min_samples_leaf,max_leaf_nodes)
            tree.fit(X_train, y_train)

            #prediction
            y_pred = tree.predict(X_test)
            labels = np.unique(y_pred)

            #matrice de confusion
            df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred,labels=labels),columns=labels, index=labels)
            df_cm.insert(0, target, df_cm.index)

            #affichage graphique des prédictions réalisé
            pca = PCA(n_components=2)
            temp = pca.fit_transform(X_test)
            coord = pd.DataFrame(temp,columns=["PCA1","PCA2"])
            Y_pred = pd.DataFrame(y_pred,columns=["tree_clusters"])
            Y_test = pd.DataFrame(y_test.values,columns=[target])

            result = pd.concat([coord,Y_pred,Y_test],axis=1)
            fig_tree = px.scatter(result, x="PCA1", y="PCA2", color="tree_clusters", hover_data=['tree_clusters'],
                             title="PCA des classes prédites par le modèle")
            fig_input_data = px.scatter(result, x="PCA1", y="PCA2", color=target, hover_data=[target],
                             title="PCA du jeu de données test")

            t2 = time.time()
            # affichage l'arbre sortie graphique
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'plot_button' in changed_id:
                plot_tree(tree,max_depth=max_depth,
                                 feature_names=feature,
                                 class_names=y.unique(),
                                 filled=True)
                plt.show()

            #html.P('Résult {}'.format(str(moy)))
            return html.Div(
                ["Matrice de confusion : ",html.Br(),
                 dash_table.DataTable(
                     id='Tree_cm',
                     columns=[{"name": i, "id": i} for i in df_cm.columns],
                     data=df_cm.to_dict('records'),),
                 html.Br(),"f1_score : {}".format(f1_score(y_test, y_pred,average="macro")),html.Br(),
                 "recall score : {}".format(recall_score(y_test, y_pred,average="macro")),
                 html.Br(),"precision score : {}".format(precision_score(y_test, y_pred,average="macro")),
                 html.Br(),"accuracy score : {}".format(accuracy_score(y_test, y_pred)),
                 html.Br(),
                 dcc.Graph(
                     id='res_Tree_FitPredict_graph',
                     figure=fig_tree),
                 dcc.Graph(
                     id='res_Tree_FitPredict_inputgraph',
                     figure=fig_input_data),
                 "temps : {} sec".format(t2-t1)]),""


def CrossValidation(app):
    # Cross Validation (Ok )
    @app.callback(
        Output(component_id='res_Tree_CrossValidation',component_property='children'),
        Output(component_id="ls-loading-output-2_tree", component_property="children"),
        Input(component_id='Tree_button_CrossValidation',component_property='n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State('criterion','value'),
        State('splitter','value'),
        State('max_depth','value'),
        State('min_samples_split','value'),
        State('min_samples_leaf','value'),
        State('max_leaf_nodes','value'),
        State(component_id='Tree_cv_number_of_folds',component_property='value'),
        State(component_id='Tree_cv_scoring',component_property='value'))
    def CV_score(n_clicks,file,target,features,criterion,splitter,max_depth,min_samples_split,min_samples_leaf,max_leaf_nodes,cv_number_of_folds,cv_scoring):
        if (n_clicks == 0):
            return "",""
        else:

            if max_depth == 0:
                max_depth = None
            if max_leaf_nodes == 0:
                max_leaf_nodes = None

            df = get_pandas_dataframe(file)

            X = df[features]
            Y= df[target]

            tree = build_tree(criterion, splitter, max_depth, min_samples_split,min_samples_leaf,max_leaf_nodes)
            #clf = DecisionTreeClassifier()
            res = cross_validation(clf=tree,X=X,Y=Y,cv=cv_number_of_folds,scoring=cv_scoring)
            #print(res[0])
            return html.Div([
                "cross validation ",html.B("{} : ".format(cv_scoring)),
                html.B(["{}".format(np.mean(res[0]))],style={'color': 'green'}),html.Br(),
                html.Br(),"temps : {} sec".format(res[1])]),""
