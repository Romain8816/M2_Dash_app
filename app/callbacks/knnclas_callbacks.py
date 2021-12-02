from dash.dependencies import Input,Output,State
from fonctions.various_functions import get_pandas_dataframe, binariser, centrer_reduire_norm, split_train_test, pre_process
from fonctions.algo_functions import build_KNeighborsClassifier, cross_validation, get_best_params
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score
from sklearn.decomposition import PCA
from dash import html, dash_table, dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import time

######################################
# Callback est en charge du calcul
# des paramètres optimaux pour le modèle
# de classification KNN
######################################
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
        State(component_id='KNeighborsClassifier_GridSearchCV_njobs',component_property='value'),
        State(component_id='KNeighborsClassifier_random_state',component_property='value'),
        State(component_id='KNeighborsClassifier_test_size',component_property='value'),
        State(component_id='KNeighborsClassifier_shuffle',component_property='value'),
        State(component_id='KNeighborsClassifier_stratify',component_property='value'))
    def GridSearchCV_score(n_clicks,file,target,features,num_variables,centrer_reduire,GridSearchCV_number_of_folds,GridSearchCV_scoring,njobs,random_state,test_size,shuffle,stratify):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time() # start
            if njobs == "None":
                njobs = None
            df = get_pandas_dataframe(file) # récupération du jeu de données
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,Y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)
            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=Y,random_state=random_state,test_size=test_size,shuffle=shuffle,stratify=stratify)
            # calcul des hyperparamètres optimaux :
            params = {'n_neighbors':list(range(1,21)), 'weights':["uniform","distance"], 'algorithm':["auto","brute"], 'leaf_size':[5,10,20,30,40], 'p':[1,2], 'metric':["minkowski","euclidean","manhattan"]} # liste des paramètres à tester (liste non exhaustive en raison du temps de calcul)
            grid_search = get_best_params(X=X_train,Y=y_train,clf="KNeighborsClassifier",params=params,cv=GridSearchCV_number_of_folds,scoring=GridSearchCV_scoring,njobs=njobs) # Optimisation des hyperparamètres avec parallélisation possible
            if isinstance(grid_search, str) == False:
                best_params = pd.Series(grid_search.best_params_,index=grid_search.best_params_.keys())
                best_params = pd.DataFrame(best_params)
                best_params.reset_index(level=0, inplace=True)
                best_params.columns = ["paramètres","valeurs"]
                t2 = time.time() # stop
                diff = t2 - t1 # calcul du temps écoulé pour la section 'optimisation des hyperparamètres'
                return html.Div(["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),dash_table.DataTable(id='KNeighborsClassifier_params_opti',columns=[{"name": i, "id": i} for i in best_params.columns],data=best_params.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in best_params.columns]),html.Br(),html.Br(),"GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{:.4f}".format(abs(grid_search.best_score_))],style={'color': 'blue'}),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff)]),""
            else:
                t2 = time.time() # stop
                diff = t2 - t1 # calcul du temps écoulé pour la section 'optimisation des hyperparamètres'
                return html.Div(["GridSearchCV paramètres optimaux : ",html.Br(),html.Br(),"{}".format(grid_search)]),""

######################################
# Callback en charge d'apprendre le modèle de regression KNN sur
# le jeu de données d'entrainement et de calculer
# ses performance sur le jeu de données test
######################################
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
        State(component_id='KNeighborsClassifier_random_state',component_property='value'),
        State(component_id='KNeighborsClassifier_test_size',component_property='value'),
        State(component_id='KNeighborsClassifier_shuffle',component_property='value'),
        State(component_id='KNeighborsClassifier_stratify',component_property='value'))
    def CV_score(n_clicks,file,target,features,num_variables,n_neighbors,weights,algorithm,leaf_size,p,metric,centrer_reduire,random_state,test_size,shuffle,stratify):
        if (n_clicks == 0):
            return "",""
        else:
            t1 = time.time() # start
            df = get_pandas_dataframe(file) # récupération du jeu de données
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,Y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)
            # split train test
            X_train,X_test,y_train,y_test = split_train_test(X=X,Y=Y,random_state=random_state,test_size=test_size,shuffle=shuffle,stratify=stratify)
            clf = build_KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric) # instanciation du modèle
            clf.fit(X_train.values,y_train.values) # apprentissage
            y_pred = clf.predict(X_test.values) # prédiction
            labels = np.unique(y_test)
            df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred,labels=labels),columns=labels, index=labels) # matrice de confusion
            df_cm.insert(0, target, df_cm.index)
            pca = PCA(n_components=2)
            temp = pca.fit_transform(X_test)
            coord = pd.DataFrame(temp,columns=["PCA1","PCA2"]) # calcul des coordonnées pour l'ACP
            Y_pred = pd.DataFrame(y_pred,columns=["knn_clusters"])
            Y_test = pd.DataFrame(y_test.values,columns=[target])
            result = pd.concat([coord,Y_pred,Y_test],axis=1)
            fig_knn = px.scatter(result, x="PCA1", y="PCA2", color="knn_clusters", hover_data=['knn_clusters'],
                             title="PCA des classes prédites par le modèle".format(file.split("/")[-1]))
            fig_input_data = px.scatter(result, x="PCA1", y="PCA2", color=target, hover_data=[target],
                             title="PCA du jeu de données test ")
            t2 = time.time() # stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'performance du modèle sur le jeu test'
            if len(set(list(Y))) > 2: # si le nombre de classe de la variable explicative est > 2 (non binaire), on renvoie les métriques pertinentes
                return html.Div(["Matrice de confusion : ",html.Br(),dash_table.DataTable(id='KNeighborsClassifier_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),html.B("f1_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(f1_score(y_test, y_pred,average="macro"),f1_score(y_test, y_pred,average="micro"),f1_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("recall_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(recall_score(y_test, y_pred,average="macro"),recall_score(y_test, y_pred,average="micro"),recall_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("precision_score "),"macro {:.4f} , micro {:.4f}, weighted {:.4f}".format(precision_score(y_test, y_pred,average="macro"),precision_score(y_test, y_pred,average="micro"),precision_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("accuracy_score ")," {:.4f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff),html.Br(),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_knngraph', figure=fig_knn),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_inputgraph', figure=fig_input_data)]),""
            else:
                return html.Div(["Matrice de confusion : ",html.Br(),dash_table.DataTable(id='KNeighborsClassifier_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),html.B("f1_score "),"binary {:.4f}".format(f1_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("recall_score "),"binary {:.4f}".format(recall_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("precision_score "),"binary {:.4f}".format(precision_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("accuracy_score "),"{:.4f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff),html.Br(),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_knngraph', figure=fig_knn),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_inputgraph', figure=fig_input_data)]),""


######################################
# Callback en charge de faire la validation
# croisée du modèle de regression KNN
######################################
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
            t1 = time.time() # start
            df = get_pandas_dataframe(file) # récupération du jeu de données
            # replacer NA par moyenne ou mode, binariser et centrer réduire
            X,Y = pre_process(df=df,num_variables=num_variables,features=features,centrer_reduire=centrer_reduire,target=target)
            clf = build_KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
            cv_res = cross_validation(clf=clf,X=X,Y=Y,cv=cv_number_of_folds,scoring=cv_scoring) # validation croisée
            t2 = time.time()# stop
            diff = t2 - t1 # calcul du temps écoulé pour la section 'validation croisée'
            if isinstance(cv_res, str) == False:
                return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.4f}".format(abs(np.mean(cv_res)))],style={'color': 'green'}),html.Br(),html.Br(),"temps : {:.4f} sec".format(diff)]),""
            else:
                return html.Div(["cross validation :",html.Br(),"{}".format(cv_res)]),""
