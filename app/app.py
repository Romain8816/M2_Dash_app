import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output,State
import os
import pandas as pd
import json
from dash.exceptions import PreventUpdate


from layout.layout import dataset_selection, target_selection,features_selection, data_path

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title="Machine Learning App"


form = dbc.Form([dataset_selection,target_selection,features_selection])

app.layout = html.Div(children=[
        
        dcc.Store(id='columns'), #va servir à stocker le nom des variables du dataset
        html.Div(
            [
                html.H1('Réalisation d’une interface d’analyse de données par apprentissage supervisé'),
                html.H5(['Olivier IMBAUD, Inès KARA, Romain DUDOIT'],style={'color':'white','font-weight':'bold'}),
                html.H6('Master SISE (2021-2022)',style={'color':'white','font-weight':'bold'})
            ],className='container-fluid top'
        ),
        html.Div(
            form, className='container-fluid'
        ),
        html.Div(id='test')
])


# Chargement des variables pour la variable cible à sélectionner selon le fichier choisit --------------------------------------------------------------------------
@app.callback(
    Output(component_id='target_selection', component_property='value'),
    Output(component_id='target_selection', component_property='options'),
    Input(component_id='file_selection', component_property='value')
)
def FileSelection(file):
    if file is None:
        raise PreventUpdate
    else:
        variables=pd.read_csv(data_path+file, header=0, nrows=0).columns.tolist()
        return (None,[{'label':v, 'value':v} for v in variables])

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


