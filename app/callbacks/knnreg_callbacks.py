from dash.dependencies import Input,Output,State
from fonctions.various_functions import get_pandas_dataframe, binariser, centrer_reduire_norm, split_train_test, pre_process
from fonctions.algo_functions import build_KNeighborsRegressor, cross_validation, get_best_params
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from dash import html, dash_table, dcc, html
import plotly.graph_objects as go
import plotly.express as px
import math
import pandas as pd
import numpy as np
import time


######################################
# Callback est en charge du calcul
# des paramètres optimaux pour le modèle
# de regression KNN
######################################
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
        State(component_id='KNeighborsRegressor_GridSearchCV_njobs',component_property='value'),
        State(component_id='KNeighborsRegressor_random_state',component_property='value'),
        State(component_id='KNeighborsRegressor_test_size',component_property='value'),
        State(component_id='KNeighborsRegressor_shuffle',component_property='value'))
    def GridSearchCV_score(n_clicks,file,target,features,num_variables,centrer_reduire,GridSearchCV_number_of_folds,GridSearchCV_scoring,njobs,random_state,test_size,shuffle):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time()  # start
            if njobs == "None":
                njobs = None
            df = get_pandas_dataframe(file)  # récupération du jeu de données
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,Y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)
            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=Y,random_state=random_state,test_size=test_size,shuffle=shuffle)
            # calcul des hyperparamètres optimaux :
            params = {'n_neighbors':list(range(1,21)), 'weights':["uniform","distance"], 'algorithm':["auto","brute"], 'leaf_size':[5,10,20,30,40], 'p':[1,2], 'metric':["minkowski","euclidean","manhattan"]} # liste des paramètres à tester (liste non exhaustive en raison du temps de calcul)
            grid_search = get_best_params(X=X_train,Y=y_train,clf="KNeighborsRegressor",params=params,cv=GridSearchCV_number_of_folds,scoring=GridSearchCV_scoring,njobs=njobs) # Optimisation des hyperparamètres avec parallélisation possible
            best_params  = pd.Series(grid_search.best_params_,index=grid_search.best_params_.keys())
            best_params  = pd.DataFrame(best_params )
            best_params .reset_index(level=0, inplace=True)
            best_params .columns = ["paramètres","valeur"]
            t2 = time.time() # stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'optimisation des hyperparamètres'
            if GridSearchCV_scoring == "RMSE":
                grid_search.best_score_ = math.sqrt(abs(grid_search.best_score_))
            return html.Div(
                    ["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),
                    dash_table.DataTable(
                        id='KNeighborsRegressor_params_opti',
                        columns=[{"name": i, "id": i} for i in best_params.columns],
                        data=best_params.to_dict('records'),
                        style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in best_params.columns]),
                        html.Br(),html.Br(),
                        "GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{:.4f}".format(abs(grid_search.best_score_))],style={'color': 'blue'}),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff)]
            ),""

######################################
# Callback en charge d'apprendre le modèle de regression KNN sur
# le jeu de données d'entrainement et de calculer
# ses performances sur le jeu de données test
######################################
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
        State(component_id='KNeighborsRegressor_random_state',component_property='value'),
        State(component_id='KNeighborsRegressor_test_size',component_property='value'),
        State(component_id='KNeighborsRegressor_shuffle',component_property='value'))
    def CV_score(n_clicks,file,target,features,num_variables,n_neighbors,weights,algorithm,leaf_size,p,metric,centrer_reduire,random_state,test_size,shuffle):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time() # start
            df = get_pandas_dataframe(file) # récupération du jeu de données
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,Y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)
            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=Y,random_state=random_state,test_size=test_size,shuffle=shuffle)

            clf = build_KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric) # instanciation du modèle
            clf.fit(X_train.values,y_train.values) # apprentissage
            y_pred = clf.predict(X_test.values) # prédiction

            k = 0
            more_uniq_col = ""
            for col in X_test: # récupérer la variable explicative avec le plus de valeurs uniques pour la représentation graphique
                if len(X_test[col].unique()) > k:
                    more_uniq_col = col
                    k = len(X_test[col].unique())

            X_test["y_test"] = y_test.tolist()
            X_test["y_pred"] = y_pred
            X_test = X_test.sort_values(by=more_uniq_col)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X_test[more_uniq_col],
                y=X_test["y_pred"],
                mode='lines+markers',
                name='y_pred',
                marker={'size': 8, "opacity":0.8}
            ))
            fig.add_trace(go.Scatter(
                x=X_test[more_uniq_col],
                y=X_test["y_test"],
                mode='markers',
                name='y_test',
                marker={'size': 8, "opacity":0.5}
            ))
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
            t2 = time.time() # stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'performance du modèle sur le jeu test'

            return html.Div(
                [
                    html.B("Carré moyen des erreurs (MSE) "),": {:.4f}".format(abs(mean_squared_error(y_test, y_pred))),html.Br(),html.Br(),
                    html.B("Erreur quadratique moyenne (RMSE) "),": {:.4f}".format(math.sqrt(abs(mean_squared_error(y_test, y_pred)))),html.Br(),html.Br(),
                    html.B("Erreur moyenne absolue (MAE) "),": {:.4f}".format(abs(mean_absolute_error(y_test, y_pred))),html.Br(),html.Br(),
                    html.B("Coefficient de détermination (R2) "),": {:.4f}".format(abs(r2_score(y_test, y_pred))),html.Br(),html.Br(),
                    "temps : {:.4f} sec".format(diff),html.Br(),html.Br(),
                    dcc.Graph(id='res_KNeighborsRegressor_FitPredict_knngraph', figure=fig),html.Br(),html.Br(),
                ]
            ),""

######################################
# Callback en charge de faire la validation
# croisée du modèle de regression KNN
######################################
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
            t1 = time.time() # start
            df = get_pandas_dataframe(file)
            df = get_pandas_dataframe(file) # récupération du jeu de données
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,Y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)
            clf = build_KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
            cv_res = cross_validation(clf=clf,X=X,Y=Y,cv=cv_number_of_folds,scoring=cv_scoring)
            t2 = time.time()# stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'validation croisée'
            return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.4f}".format(abs(np.mean(cv_res)))],style={'color': 'green'}),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff)]),""
