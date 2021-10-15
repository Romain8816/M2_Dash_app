import dash
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
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
        html.H1('Machine Learning App',className='h1'),

        html.P('Application réalisée par Olivier IMBAUD, Inès KARA, Romain DUDOIT'),
        dcc.Dropdown(
            id='file_selection',
            options=[
                {'label':i, 'value':i} for i in files
            ],
            searchable=False,
            placeholder="Choisir un jeu de données",
            clearable=False
        ), 

        html.Div([
            dcc.Dropdown(id='variable_cible', placeholder="Sélectionner la variable cible", searchable=False, style={'display':'none'}),
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
        return {'display': 'block'}, [{'label':v, 'value':v} for v in variable_list]


if __name__=='__main__':
    app.run_server(debug=True)