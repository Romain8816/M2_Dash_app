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
#import dask.dataframe as dd

from layout.layout import drag_and_drop, parse_contents, location_folder, dataset_selection, target_selection,features_selection, data_path

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title="Machine Learning App"


form = dbc.Form([location_folder, dataset_selection,target_selection,features_selection])

regression_models = ['Régression linéaire', 'Régression polynomiale', 'Régression lasso']
classfication_models = ['Arbre de décision','SVM','KNN',"CAH","kmeans"]

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


# @app.callback(
#     Output('',''),
#     Input('submit-button-state', 'n_clicks'),
#     State(component_id="location_folder",component_property='value')
# )




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





# Chargement des variables pour la variable cible à sélectionner selon le fichier choisit --------------------------------------------------------------------------
@app.callback(
    Output(component_id='target_selection', component_property='value'),
    Output(component_id='target_selection', component_property='options'),
    Output(component_id='dataset', component_property='children'),
    Output(component_id='num_variables', component_property='data'),
    Input(component_id='file_selection', component_property='value')
)
def FileSelection(file):
    if file is None:
        raise PreventUpdate
    else:
        with open(data_path+file) as myfile:
            firstline = myfile.readline()
        myfile.close()
        deliminter = detect(firstline)

        df = pd.read_csv(data_path+file,sep=deliminter)
        variables = df.columns.tolist()
        num_variables = df.select_dtypes(include=np.number).columns.tolist()
        table =dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name":i,"id":i} for i in df.columns],
                #fixed_rows={'headers': True},
                page_size=20,
                style_cell={'textAlign': 'left'},
                style_table={'height': '400px', 'width': '100%', 'minwidth': '100%', 'overflowY': 'auto','overflowX': "scroll"},
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
