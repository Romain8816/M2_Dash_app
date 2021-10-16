import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.html.Div import Div
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output
import os
import pandas as pd

app = dash.Dash(__name__)
app.title="Machine Learning App"

data_path = os.getcwd() +'\\data\\'
files = [f for f in os.listdir(data_path)]

app.layout = html.Div(children=[
        # 
        html.Div(
            [
                html.H1('Réalisation d’une interface d’analyse de données par apprentissage supervisé'),
                html.H5(['Olivier IMBAUD, Inès KARA, Romain DUDOIT'],style={'color':'white','font-weight':'bold'}),
                html.H6('Master SISE (2021-2022)',style={'color':'white','font-weight':'bold'})
            ],className='container-fluid top'
        ),
        html.Div(
            [
                html.Label("Jeu de données"),
                dcc.Dropdown(
                    id='file_selection',
                    options=[{'label':i, 'value':i} for i in files],
                    searchable=False,
                    placeholder="Choisir un jeu de données",
                    clearable=False, 
                    style={'width':'50%'},
                )
            ],className='container-fluid'
        ),
        html.Div([
            dcc.Dropdown(id='variable_cible', placeholder="Sélectionner la variable cible", searchable=False, style={'display':'none'})
        ])
        
])

@app.callback(
    Output(component_id='variable_cible', component_property='style'),
    Output(component_id='variable_cible', component_property='options'),
    Input(component_id='file_selection', component_property='value')
)
def selection_variable_cible(input_value):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné
    if input_value is None:
        raise PreventUpdate
    else :
        df = pd.read_csv(data_path+input_value)
        variable_list= list(df.columns.values)
        return {'display': 'block', 'width':'50%'}, [{'label':v, 'value':v} for v in variable_list]


if __name__=='__main__':
    app.run_server(debug=True)