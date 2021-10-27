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
from detect_delimiter import detect
import dash_daq as daq
import cchardet as chardet
from scipy.sparse.construct import rand, random
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

from layout.layout import drag_and_drop, location_folder, dataset_selection, target_selection,features_selection, kmeans_params_and_results
from layout.layout import regression_tabs, classification_tabs
from fonctions.various_functions import allowed_files, get_pandas_dataframe, parse_contents
from fonctions.algo_functions import build_kmeans
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
app.title="Machine Learning App"


# VARIABLES 
form = dbc.Form([location_folder, dataset_selection,target_selection,features_selection])
form_kmeans_params_and_results = dbc.Form([kmeans_params_and_results])

regression_models = ['Régression linéaire', 'Régression polynomiale', 'Régression lasso']
classification_models = ['Arbre de décision','SVM','KNN',"CAH","kmeans"]
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
        drag_and_drop,
        html.Div(
            [
                dbc.Row(
                    [
                        form,
                        html.Br(),
                        dbc.Col(html.Div(id='dataset'),width="100%"),
                        html.P(id='nrows',children="Nombre d'observation : ",className="mb-3"),
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
        form_kmeans_params_and_results,
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
        print(variables)
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
# (SVM) 

@app.callback(
    Output('res_svm','children'),
    Input('smv_button','n_clicks'),
    State(component_id='file_selection',component_property='value'),
    State(component_id='target_selection',component_property='value'),
    State(component_id='features_selection',component_property='value'),
    State(component_id='test_size',component_property='value'),
    State(component_id='random_state',component_property='value'),
    State(component_id='k_fold',component_property='value'),
    State(component_id='svm_kernel_selection',component_property='value'),          # Noyau
    State(component_id='svm_regularisation_selection',component_property='value'),  # C
    State(component_id='svm_epsilon',component_property='value'))

def score (n_clicks,file,target,features,test_size,random_state,k_fold,kernel,regularisation,epsilon):

    if (n_clicks == 0):
        PreventUpdate
    else:
        df = get_pandas_dataframe(file)

        X= df[features]
        y= df[target]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

        numerical_features = make_column_selector(dtype_include=np.number)
        categorical_features = make_column_selector(dtype_exclude=np.number)

        categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(drop='first',sparse=False))
        numerical_pipeline = make_pipeline(SimpleImputer(),StandardScaler())

        preprocessor = make_column_transformer((numerical_pipeline,numerical_features),
                                                (categorical_pipeline,categorical_features))

        model = make_pipeline(preprocessor,svm.SVR(kernel=kernel,C=regularisation,epsilon=epsilon))
        score = cross_val_score(model,X_train,y_train)

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        
        return html.P("R² moyen : "+ str(score.mean()) )









########################################################################################################
# Affichage des paramètres du modèle (pour le moment uniquement kmeans)
# @app.callback(
#     Output(component_id='kmeans-container',component_property='style'),
#     Output(component_id='n_clusters',component_property='value'),
#     Input(component_id='model_selection',component_property='value'),
#     Input(component_id='file_selection', component_property='value'),
#     Input(component_id='target_selection',component_property='value'))
# def ModelParameters(model,file_path,target):
#     if file_path is None:
#         raise PreventUpdate
#     else:
#         df = get_pandas_dataframe(file_path)
#         if model == "kmeans":
#             return {"margin":25,"display":"block"},len(set(list(df[target])))
#         else:
#             raise PreventUpdate

# @app.callback(
#     Output(component_id='kmeans-explore-object',component_property='options'),
#     Output(component_id='kmeans-explore-object',component_property='value'),
#     Input(component_id='model_selection',component_property='value'),
#     Input(component_id='file_selection', component_property='value'),
#     Input(component_id='target_selection',component_property='value'),
#     Input(component_id='features_selection',component_property='value'),
#     Input(component_id='n_clusters',component_property='value'),
#     Input(component_id='init',component_property='value'),
#     Input(component_id='n_init',component_property='value'),
#     Input(component_id='max_iter',component_property='value'),
#     Input(component_id='tol',component_property='value'),
#     Input(component_id='verbose',component_property='value'),
#     Input(component_id='random_state',component_property='value'),
#     Input(component_id='algorithm',component_property='value'),
#     Input(component_id='centrer_reduire',component_property='value'),
#     Input('num_variables','data'))

# def ShowModelAttributes(model,file_path,target,features,n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire,num_variables):
#     if file_path is None:
#         raise PreventUpdate
#     else:
#         df = get_pandas_dataframe(file_path)
#         if model == "kmeans":

#             if any(item not in num_variables for item in features) == True:
#                 df_ = pd.get_dummies(df.loc[:, df.columns != target])
#                 features = list(df_.columns)
#                 df = pd.concat([df_,df[target]],axis=1)

#             if random_state == "None":
#                 random_state = None

#             kmeans = build_kmeans(df[features],n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire)
#             return [{"label":v,"value":v} for v in list(kmeans.__dict__.keys())+["randscore_"] if v.endswith("_")],"randscore_"
#         else:
#             raise PreventUpdate

# @app.callback(
#     Output(component_id='kmeans-explore-object-display',component_property='children'),
#     Output(component_id='kmeans-pca',component_property='figure'),
#     Output(component_id='input-pca',component_property='figure'),
#     Input(component_id='model_selection',component_property='value'),
#     Input(component_id='file_selection', component_property='value'),
#     Input(component_id='target_selection',component_property='value'),
#     Input(component_id='features_selection',component_property='value'),
#     Input(component_id='n_clusters',component_property='value'),
#     Input(component_id='init',component_property='value'),
#     Input(component_id='n_init',component_property='value'),
#     Input(component_id='max_iter',component_property='value'),
#     Input(component_id='tol',component_property='value'),
#     Input(component_id='verbose',component_property='value'),
#     Input(component_id='random_state',component_property='value'),
#     Input(component_id='algorithm',component_property='value'),
#     Input(component_id='centrer_reduire',component_property='value'),
#     Input(component_id='kmeans-explore-object',component_property='value'),
#     Input('num_variables','data'))
# def ShowModelResults(model,file_path,target,features,n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire,kmeans_object_value,num_variables):
#     if file_path is None:
#         raise PreventUpdate
#     else:
#         df = get_pandas_dataframe(file_path)
#         if model == "kmeans":
#             if any(item not in num_variables for item in features) == True:
#                 df_ = pd.get_dummies(df.loc[:, df.columns != target])
#                 features = list(df_.columns)
#                 df = pd.concat([df_,df[target]],axis=1)

#             if random_state == "None":
#                 random_state = None
#             kmeans = build_kmeans(df[features],n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire)
#             #y = list(df[target].replace({"setosa":0,"versicolor":1,"virginica":2}))
#             setattr(kmeans, 'randscore_', adjusted_rand_score(kmeans.labels_,df[target]))
#             pca = PCA(n_components=2)
#             temp = pca.fit_transform(df[features])
#             coord = pd.DataFrame(temp,columns=["PCA1","PCA2"])
#             Y_pred = pd.DataFrame(list(map(str,kmeans.labels_)),columns=["kmeans_clusters"])
#             result = pd.concat([coord,Y_pred,df[target]], axis=1)
#             fig_kmeans = px.scatter(result, x="PCA1", y="PCA2", color="kmeans_clusters", hover_data=['kmeans_clusters'],
#                              title="PCA du jeu de données {} colorié par clusters du KMeans".format(file_path.split("/")[-1]))
#             fig_input_data = px.scatter(result, x="PCA1", y="PCA2", color=target, hover_data=[target],
#                              title="PCA du jeu de données {} colorié en fonction de la variable à prédire".format(file_path.split("/")[-1]))
#             return html.P("{}".format(getattr(kmeans, kmeans_object_value))),fig_kmeans,fig_input_data
#         else:
#             raise PreventUpdate

# @app.callback(
#     Output('test','children'),
#     Input('file_selection','value'),
#     Input('target_selection','value'),
#     Input('features_selection','value')
# )
# def display(file_selection,target_selection,feature_selection):
#     ctx=dash.callback_context

#     ctx_msg = json.dumps({
#         'states': ctx.states,
#         'triggered': ctx.triggered,
#         'inputs': ctx.inputs
#     }, indent=2)

#     return html.Div([
#         html.Pre(ctx_msg)
#     ])

# Affichage du tableau après ajout d'un fichier.
# @app.callback(Output('output-data-upload', 'children'),
#               Input('upload-data', 'contents'), # les données du fichier
#               State('upload-data', 'filename'), # nom du fichier
# )
# def update_output(list_of_contents, list_of_names):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n) for c, n in
#             zip(list_of_contents, list_of_names)]
#         return children

app.css.append_css({'external_url': './assets/style2.css' # LINUX - MAC-OS
})

if __name__=='__main__':
    app.run_server(debug=True)
