from tkinter.constants import NONE
import dash
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
import base64
import io
import plotly.graph_objects as go
import time
from detect_delimiter import detect
import dash_daq as daq
import cchardet as chardet
from scipy.sparse.construct import rand, random
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import validation_curve

from layout.layout import location_folder, dataset_selection, target_selection,features_selection
from layout.layout import regression_tabs, classification_tabs
from fonctions.various_functions import allowed_files, get_pandas_dataframe, parse_contents
from fonctions.algo_functions import build_smv
from plotly import tools as tls
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from statistics import *
from layout.layout import location_folder, dataset_selection, target_selection,features_selection
from layout.layout import regression_tabs, classification_tabs
from fonctions.various_functions import allowed_files, get_pandas_dataframe, parse_contents
from fonctions.algo_functions import *
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score
from math import log, sqrt
from matplotlib import pyplot

from callbacks import svr_callbacks, log_callbacks

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
app.title="Machine Learning App"


# VARIABLES
form = dbc.Form([location_folder, dataset_selection,target_selection,features_selection])

allowed_extensions =('.csv','.xlsx','.xls')



#************************************************************************ MAIN LAYOUT **********************************************************************
#***********************************************************************************************************************************************************
app.layout = html.Div(children=[
        html.Div(
            [
                html.H1('Réalisation d’une interface d’analyse de données par apprentissage supervisé'),
                html.H5(['Olivier IMBAUD, Inès KARA, Romain DUDOIT'],style={'color':'white','font-weight':'bold'}),
                html.H6('Master SISE (2021-2022)',style={'color':'white','font-weight':'bold'})
            ],className='container-fluid top'
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        form,
                        html.Br(),
                        dbc.Col(html.Div(id='dataset'),width="100%"),
                        html.P(id='nrows',children="",className="mb-3"),
                    ]
                )
            ], className='container-fluid'
        ),
        #html.Div(id='output-data-upload'), # Affichage du tableau
        html.Div(id='stats'),
        #dcc.Graph(id='stats'),
        html.Div(
            dbc.Checklist(
                id="centrer_reduire"
            ),
            className='container-fluid'
        ),
        html.Div(
            [
                # Affichage des tabs, caché par défaut
                dbc.Collapse(
                    id='collapse_tab',
                    is_open=False
                ),

                dbc.RadioItems(
                    id="model_selection",
                ),
            ],
            className='container-fluid'
        ),
        html.Br(),
        html.Br(),
        dcc.Store(id='num_variables')
])

#*********************************************************************** CALLBACKS *************************************************************************
#***********************************************************************************************************************************************************

########################################################################################################
# (INIT) RECUPERATION DE LA LISTE DES FICHIERS AUTORISES DANS LE REPERTOIRE RENSEIGNE

@app.callback(
    Output('file_selection','options'), # mise à jour de la liste des fichiers dans le répertoire
    Input('validation_folder', 'n_clicks'), # valeur du bouton
    State(component_id="location_folder",component_property='value') #valeur de l'input
)
def update_files_list(n_clicks,data_path):
    # Si on a appuyer sur le bouton valider alors
    if n_clicks !=0:
        # On essaie de parcourir les fichiers dans le répertoire data_path
        try :
            files = os.listdir(r'%s' %data_path)
            filtred_files=allowed_files(data_path,allowed_extensions)
        # Si le répertoire n'existe
        except:
            return dash.no_update, '{} is prime!'.format(data_path)    #/!\ Exception à reprendre

        # --- /!\ return ([{'label':f, 'value':(r'%s' %(data_path+'\\'+f))} for f in filtred_files]) # WINDOWS
        return ([{'label':f, 'value':(r'%s' %(data_path+'/'+f))} for f in filtred_files]) # LINUX / MAC-OS
    else:
        raise PreventUpdate

########################################################################################################
# (INIT) LECTURE DU FICHIER CHOISIT ET MAJ DE LA DROPDOWN DES VARIABLES CIBLES

@app.callback(
    Output(component_id='target_selection', component_property='value'),
    Output(component_id='target_selection', component_property='options'),
    Output(component_id='dataset', component_property='children'),
    Output(component_id='num_variables', component_property='data'),
    Input(component_id='file_selection', component_property='value'),
)
def FileSelection(file_path):
    if file_path is None:
        raise PreventUpdate
    else:
        df = get_pandas_dataframe(file_path)
        variables = df.columns.tolist()
        num_variables = df.select_dtypes(include=np.number).columns.tolist()
        table =dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name":i,"id":i} for i in df.columns],
                fixed_rows={'headers': True},
                page_size=20,
                sort_action='native',
                sort_mode='single',
                sort_by=[],
                style_cell={'textAlign': 'left','minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_table={'height': '400px', 'overflowY': 'scroll','overflowX': 'scroll'},
                style_header={'backgroundColor': 'dark','fontWeight': 'bold'},
                style_cell_conditional=[
                    {'if': {'column_id': c},'textAlign': 'center'} for c in df.columns],
            )
        return (None,[{'label':v, 'value':v} for v in variables],table,num_variables)

########################################################################################################
# (INIT) CHARGEMENT DES VARIABLES EXPLICATIVES A SELECTIONNER

@app.callback(
        Output(component_id='features_selection', component_property='options'),
        Output(component_id='features_selection', component_property='value'),
        #Output(component_id='collapse_tab', component_property='is_open'),
        Input(component_id='target_selection', component_property='value'),   # valeur de la variable cible
        Input(component_id='target_selection', component_property='options'), # liste des variables cibles
        Input(component_id='features_selection', component_property='value')  # valeur des variables explicatives.
)
def TargetSelection(target,options,feature_selection_value):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné
    if target is None:
        return ([{'label':"", 'value':""}],None)
    else :
        variables = [d['value'] for d in options]
        if feature_selection_value == None:
            return (
                [{'label':v, 'value':v} for v in variables if v!=target],
                [v for v in variables if v!=target]
            )
        else:
            if len(feature_selection_value) >= 2:
                return (
                    [{'label':v, 'value':v} for v in variables if v!=target],
                    [v for v in feature_selection_value if v!=target]
                )
            else:
                return (
                    [{'label':v, 'value':v} for v in variables if v!=target],
                    [v for v in variables if v!=target]
                )

########################################################################################################
# (INIT) Proposition du/des modèles qu'il est possible de sélectionner selon le type de la variable cible

@app.callback(
    #Output(component_id='model_selection',component_property='options'),
    Output(component_id='centrer_reduire',component_property='options'),
    Output(component_id='collapse_tab',component_property='children'),    # Tab de classification ou de régression
    Output(component_id='collapse_tab',component_property='is_open'),     # Affichage des onglets
    Input(component_id='file_selection', component_property='value'),     # Emplacement du fichier
    Input(component_id='num_variables',component_property='data'),        # Liste des variables numérique
    Input(component_id='target_selection',component_property='value'),    # Variable cible
    Input(component_id='features_selection',component_property='value'),  # Variables explicatives
    Input(component_id='model_selection',component_property='value')      # Model choisit.
)
def ModelSelection(file,num_variables,target_selection,feature_selection,selected_model):
    # Si la variable cible à été sélectionné
    if target_selection != None:
        # Si la variable est numérique
        if target_selection in num_variables:
            return (
                [{"label":"centrer réduire","value":"yes"}],
                regression_tabs,
                True
            )

        # Sinon (si la variable est qualitative)
        else:
            return (
                [{"label":"centrer réduire","value":"yes"}],
                classification_tabs,
                True
            )
    # Sinon ne rien faire
    else:
        return ([],"",False)
        #raise PreventUpdate


########################################################################################################
# (Stats descriptives)

@app.callback(
    Output('stats','children'),
    Input('file_selection','value'),
    Input('features_selection','value'),
    Input('target_selection','value'),
    Input('num_variables','data'),
)
def stats_descrip(file,features,target,num_var):
    if None in (file,features,target):
        PreventUpdate
    else:
        df = get_pandas_dataframe(file)
        #X= df[features]
        #y= df[target]
        if target not in num_var :
            return dcc.Graph(
                figure = {
                    'data':[
                        {'x':df[target].value_counts().index.tolist(), 'y':df[target].value_counts().values.tolist(),'type': 'bar'}
                    ],
                    'layout': {
                        'title': 'Distribution de la variable '+target
                    }
                }
            )
        else :
            fig = px.histogram(df,x=target)
            return dcc.Graph(figure=fig)



########################################################################################################
# (Régression) SVM
svr_callbacks.Gridsearch(app)
svr_callbacks.fit_predict(app)

########################################################################################################
# (Classification) Régression logistique
log_callbacks.Gridsearch(app)


########################################################################################################
# (Régression) KNeighborsRegressor

# GridSearchCV
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
        params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 'weights':["uniform","distance"], 'algorithm':["auto","brute"], 'leaf_size':[5,10,20,30,40], 'p':[1,2], 'metric':["minkowski","euclidean","manhattan"]}
        grid_search = get_best_params(X=X,Y=Y,clf="KNeighborsRegressor",params=params,cv=GridSearchCV_number_of_folds,scoring=GridSearchCV_scoring,njobs=njobs)
        t2 = time.time()
        diff = t2 - t1
        return html.Div(
            ["GridSearchCV paramètres optimaux : {}".format(grid_search.best_params_),html.Br(),html.Br(),
            "GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{:.2f}".format(grid_search.best_score_)],style={'color': 'blue'}),html.Br(),html.Br(),"time : {:.2f} sec".format(diff)]),""

# FitPredict
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

        #print(pd.concat([,y_pred],axis=1))
        #fig_y_pred = px.scatter(x=, y=,color_discrete_sequence=['blue'],opacity=0.5)
        #fig_y_test = px.scatter(x=X_test.iloc[:,0], y=y_test,color_discrete_sequence=['red'],opacity=0.5)
        #fig_all = go.Figure(data=fig_y_pred.data + fig_y_test.data, name="Name of Trace 2")

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
                         html.B("Coéfficient de détermination (R2) "),": {:.2f}".format(r2_score(y_test, y_pred)),html.Br(),html.Br(),
                         "temps : {:.2f} sec".format(diff),html.Br(),html.Br(),
                         dcc.Graph(id='res_KNeighborsRegressor_FitPredict_knngraph', figure=fig),html.Br(),html.Br(),
                         ]),""


# CrossValidation
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
        return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.2f}".format(mean(res))],style={'color': 'green'}),html.Br(),html.Br(),"time : {:.2f} sec".format(diff)]),""

########################################################################################################
# (Classification) Régression logistique





# (Classification) KNeighborsClassifier

# GridSearchCV
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
        params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 'weights':["uniform","distance"], 'algorithm':["auto","brute"], 'leaf_size':[5,10,20,30,40], 'p':[1,2], 'metric':["minkowski","euclidean","manhattan"]}
        grid_search = get_best_params(X=X,Y=Y,clf="KNeighborsClassifier",params=params,cv=GridSearchCV_number_of_folds,scoring=GridSearchCV_scoring,njobs=njobs)
        t2 = time.time()
        diff = t2 - t1
        if isinstance(grid_search,str):
            return html.Div(["GridSearchCV paramètres optimaux : {}".format(grid_search),html.Br(),html.Br(),"GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{}".format(grid_search)],style={'color': 'red'}),html.Br(),html.Br(),"temps : {:.2f} sec".format(diff)]),""
        else:
            return html.Div(["GridSearchCV paramètres optimaux : {}".format(grid_search.best_params_),html.Br(),html.Br(),"GridSearchCV meilleur ",html.B(" {} ".format(GridSearchCV_scoring)),": ",html.B(["{:.2f}".format(grid_search.best_score_)],style={'color': 'blue'}),html.Br(),html.Br(),"time : {:.2f} sec".format(diff)]),""

# FitPredict
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
        if len(set(list(Y))) > 2:
            return html.Div(["Matrice de confusion : ",html.Br(),dash_table.DataTable(id='KNeighborsClassifier_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),html.B("f1_score "),"macro {:.2f} , micro {:.2f}, weighted {:.2f}".format(f1_score(y_test, y_pred,average="macro"),f1_score(y_test, y_pred,average="micro"),f1_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("recall_score "),"macro {:.2f} , micro {:.2f}, weighted {:.2f}".format(recall_score(y_test, y_pred,average="macro"),recall_score(y_test, y_pred,average="micro"),recall_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("precision_score "),"macro {:.2f} , micro {:.2f}, weighted {:.2f}".format(precision_score(y_test, y_pred,average="macro"),precision_score(y_test, y_pred,average="micro"),precision_score(y_test, y_pred,average="weighted")),html.Br(),html.Br(),html.B("accuracy_score ")," {:.2f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"time : {:.2f} sec".format(diff),html.Br(),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_inputgraph', figure=fig_input_data),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_knngraph', figure=fig_knn)]),""
        else:
            return html.Div(["Matrice de confusion : ",html.Br(),dash_table.DataTable(id='KNeighborsClassifier_cm',columns=[{"name": i, "id": i} for i in df_cm.columns],data=df_cm.to_dict('records'),style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_cm.columns],),html.Br(),html.B("f1_score "),"binary {:.2f}".format(f1_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("recall_score "),"binary {:.2f}".format(recall_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("precision_score "),"binary {:.2f}".format(precision_score(y_test, y_pred,average="binary",pos_label = sorted(list(set(list(Y))))[0])),html.Br(),html.Br(),html.B("accuracy_score "),"{:.2f}".format(accuracy_score(y_test, y_pred)),html.Br(),html.Br(),"time : {:.2f} sec".format(diff),html.Br(),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_inputgraph', figure=fig_input_data),dcc.Graph(id='res_KNeighborsClassifier_FitPredict_knngraph', figure=fig_knn)]),""

# CrossValidation
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
        if isinstance(res,str):
            return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{}".format(res)],style={'color': 'red'}),html.Br(),html.Br(),"time : {:.2f} sec".format(diff)]),""
        else:
            return html.Div(["cross validation ",html.B("{} : ".format(cv_scoring)),html.B(["{:.2f}".format(mean(res))],style={'color': 'green'}),html.Br(),html.Br(),"time : {:.2f} sec".format(diff)]),""



app.css.append_css({'external_url': './assets/style4columns.css' # LINUX - MAC-OS
})

if __name__=='__main__':
    app.run_server(debug=True)
