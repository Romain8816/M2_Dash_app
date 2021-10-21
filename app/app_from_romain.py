import dash
from dash import dcc
from dash import html
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Row import Row
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
#import dask.dataframe as dd

from layout.layout import drag_and_drop, parse_contents, location_folder, dataset_selection, target_selection,features_selection

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title="Machine Learning App"


form = dbc.Form([location_folder, dataset_selection,target_selection,features_selection])

regression_models = ['Régression linéaire', 'Régression polynomiale', 'Régression lasso']
classfication_models = ['Arbre de décision','SVM','KNN',"CAH","kmeans"]



def allowed_files(path,extensions):
    allowed_files=[]
    for file in os.listdir(path):
        if file.endswith(extensions):
            allowed_files.append(file)
        else:
            continue
    return allowed_files

    

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
                        dbc.Col(
                                html.Div(id='dataset'),width="100%"
                            )
                    ]
                )
            ]
            , className='container-fluid'
        ),
        html.Div(id='output-data-upload'), # Affichage du tableau
        html.Div(
            dbc.RadioItems(
                id="model_selection",
            ),
            className='container-fluid'
        ),
        html.Br(),
        html.Div(
            dbc.Checklist(
                id="centrer_reduire"
            ),
            className='container-fluid'
        ),
        html.Br(),
        html.Div(
            dcc.Slider(
            id='kmeans-nb-clust'
        ),
        className='container-fluid'
        ),
        dcc.Store(id='num_variables')
])


# Récupération de la liste des fichiers autorisés dans un répertoire renseigné par l'utilisateur---------------------------------------------------------------------------
@app.callback(
    Output('file_selection','options'), # mise à jour de la liste des fichiers dans le répertoire
    Input('validation_folder', 'n_clicks'), # valeur du bouton 
    State(component_id="location_folder",component_property='value') #valeur de l'input
)
def update_files_list(n_clicks,data_path):
    allowed_extensions =('.csv','.xlsx','.xls')
    # Si on a appuyer sur le bouton valider alors
    if n_clicks !=0:
        # On essaie de parcourir les fichiers dans le répertoire data_path
        try :
            files = os.listdir(r'%s' %data_path)
            filtred_files=allowed_files(data_path,allowed_extensions)
        # Si le répertoire n'existe
        except:
            return dash.no_update, '{} is prime!'.format(data_path)     ######################################/!\ Exception à reprendre

        return ([{'label':f, 'value':(r'%s' %(data_path+'\\'+f))} for f in filtred_files])
    else:
        raise PreventUpdate



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





# Lecture du fichier choisit et mise à jour de la dropdown des variables cibles possibles --------------------------------------------------------------------------
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
        if file_path.endswith('.csv'):
            with open(r'%s' %file_path, "rb") as f:
                msg = f.read()
                firstline = f.readline()
                detection = chardet.detect(msg)
                encoding= detection["encoding"]
            f.close()

            with open(r'%s' %file_path) as f:
                delimiter = detect(f.readline())
                print(delimiter)
            f.close()
        
            df = pd.read_csv(file_path,encoding=encoding,sep=delimiter)

        elif file_path.endswith(('.xls','.xlsx')):
            df = pd.read_excel(file_path)       

        variables = df.columns.tolist()
        num_variables = df.select_dtypes(include=np.number).columns.tolist()
        table =dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name":i,"id":i} for i in df.columns],
                fixed_rows={'headers': True},
                page_size=20,
                style_cell={'textAlign': 'left','minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_table={'height': '400px', 'overflowY': 'scroll','overflowX': 'scroll'},
                style_header={'backgroundColor': 'dark','fontWeight': 'bold'}
            )
        return (None,[{'label':v, 'value':v} for v in variables],table,num_variables)

# Chargement des variables pour les variables explicatives à sélectionner ---------------------------------------------------------
@app.callback(
        Output(component_id='features_selection', component_property='options'),
        Output(component_id='features_selection', component_property='value'),
        Input(component_id='target_selection', component_property='value'),
        Input(component_id='target_selection', component_property='options'),
        Input(component_id='features_selection', component_property='value')
)
def TargetSelection(target,options,feature_selection_value):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné -------------------------------------------------------------
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


# Proposition du/des modèles qu'il est possible de sélectionner selon le type de la variable cible
@app.callback(
    Output('model_selection','options'),
    Output('centrer_reduire','options'),
    Input('num_variables','data'),
    Input('target_selection','value'),
    Input('features_selection','value'),
    Input(component_id='file_selection', component_property='value'),
    Input('model_selection','value'))
def model_selection(num_variables,target_selection,feature_selection,file,selected_model):
    if target_selection != None:
        if target_selection in num_variables:
                return [{"label":v,"value":v} for v in regression_models],[{"label":"centrer réduire","value":"yes"}]
        else:
            if selected_model == "Arbre de décision":
                return [{"label":v,"value":v} for v in classfication_models],[]
            else:
                return [{"label":v,"value":v} for v in classfication_models],[{"label":"centrer réduire","value":"yes"}]
    else:
        raise PreventUpdate

@app.callback(
    Output(component_id='kmeans-nb-clust', component_property='value'),
    Input(component_id='centrer_reduire', component_property='value'),
    Input(component_id='model_selection', component_property='value'),
    Input('target_selection','value'),
    Input('features_selection','value'),
    Input(component_id='file_selection', component_property='value')
)
def param_selection(norm,model,target,features,file):

    if file != None:

        with open(data_path+file) as myfile:
            firstline = myfile.readline()
        myfile.close()
        deliminter = detect(firstline)

        df = pd.read_csv(data_path+file,sep=deliminter)

        print(target)
        print(features)

        if model_selection == "kmeans":
            pass




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


if __name__=='__main__':
    app.run_server(debug=True)
