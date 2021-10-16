import dash
from dash import dcc
from dash import html
from dash.html.Label import Label
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.html.Div import Div
from pandas.core.indexes import multi
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output
import os
import pandas as pd
import json


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
        )
])


######################################################## 
@app.callback(
    Output(component_id='columns', component_property='data'),
    Input(component_id='file_selection', component_property='value')
)
def FileSelection(file):
    if file is None:
        raise PreventUpdate
    else:
        variables=pd.read_csv(data_path+file, index_col=0, nrows=0).columns.tolist()
        return json.dumps(variables)


## Chargement des variables pour la variable cible à sélectionner
@app.callback(
    Output(component_id='target_selection', component_property='options'),
    Input(component_id='file_selection', component_property='value')
)
def TargetSelection(file):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné
    if file is None:
        raise PreventUpdate
    else :
        target_list = pd.read_csv(data_path+file, header =0, nrows=0).columns.tolist()
        return [{'label':v, 'value':v} for v in target_list]

## Chargement des variables pour les variables explicatives à sélectionner
@app.callback(
    Output(component_id='features_selection', component_property='options'),
    [
        Input(component_id='target_selection', component_property='value'),
        Input(component_id='columns', component_property='data')
    ]
)
def FeaturesSelection(target,columns):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné
    if target is None:
        raise PreventUpdate
    else :
        #target_list = pd.read_csv(data_path+file, index_col=0, nrows=0).columns.tolist()
        data = json.loads(columns)
        #df = pd.read_csv(data_path+file)
        #variable_list= list(df.columns.values)
        return ([{'label':v, 'value':v} for v in data if v!=target])

@app.callback(
    Output(component_id='features_selection', component_property='value'),
    Input(component_id='features_selection', component_property='options')
    
)
def SetDefaultValue(options):
    defaultval = [d['value'] for d in options]
    return defaultval


if __name__=='__main__':
    app.run_server(debug=True)