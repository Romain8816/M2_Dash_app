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
#import dask.dataframe as dd

from layout.layout import drag_and_drop, parse_contents, location_folder, dataset_selection, target_selection,features_selection, data_path

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title="Machine Learning App"


form = dbc.Form([location_folder, dataset_selection,target_selection,features_selection])

regression_models = ['Régression linéaire', 'Régression polynomiale', 'Régression lasso']
classfications_models = ['Arbre de décision','SVM','KNN']

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
                                html.Div(id='dataset'),width=6
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
        df = pd.read_csv(data_path+file)
        variables = df.columns.tolist()
        num_variables = df.select_dtypes(include=np.number).columns.tolist()
        table =dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name":i,"id":i} for i in df.columns],
                fixed_rows={'headers': True},
                page_size=20,
                style_cell={'textAlign': 'left'},
                style_table={'height': '400px', 'overflowY': 'auto','overflowX': 'auto'},
                style_header={'backgroundColor': 'dark','fontWeight': 'bold'}
            )
        return (None,[{'label':v, 'value':v} for v in variables],table,num_variables)

# Chargement des variables pour les variables explicatives à sélectionner ---------------------------------------------------------
@app.callback(
        Output(component_id='features_selection', component_property='options'),
        Output(component_id='features_selection', component_property='value'),
        Input(component_id='target_selection', component_property='value'),
        Input(component_id='target_selection', component_property='options')
)
def TargetSelection(target,options):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné -------------------------------------------------------------
    if target is None:
        return ([{'label':"", 'value':""}],None)
    else :
        variables = [d['value'] for d in options]
        return (
            [{'label':v, 'value':v} for v in variables if v!=target],
            [v for v in variables if v!=target]
        )

# Proposition du/des modèles qu'il est possible de sélectionner selon le type de la variable cible
# @app.callback(
#     Output('model_selection','options'),
#     Input('features_selection','value'),
# )
# def model_selection(num_variables):
#     return None


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