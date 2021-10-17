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

data_path = os.getcwd() +'\\data\\'
files = [f for f in os.listdir(data_path)]


dataset_selection = dbc.Row(
    [
        dbc.Label("", html_for="file_selection", width=1,style={'font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                id='file_selection',
                options=[{'label':i, 'value':i} for i in files],
                searchable=False,
                placeholder="Choisir un jeu de données",
                clearable=False, 
                style={'width':'50%'},
                persistence =False
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

target_selection = dbc.Row(
    [
        dbc.Label("Variable cible", html_for="target_selection", width=1,style={'color': 'red','font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                id='target_selection', 
                placeholder="Sélectionner la variable cible", 
                searchable=False,clearable=False,
                style={'width':'50%'},
                persistence=False,
                persistence_type='memory'
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

features_selection = dbc.Row(
    [
        dbc.Label("Variables explicatives", html_for="features_selection", width=1,style={'color': 'blue','font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                    id='features_selection',
                    searchable=False,
                    placeholder="Sélectionner les variables explicatives",
                    clearable=False,
                    multi=True,
                    style={'width':'50%'},
                    persistence=False,
                    persistence_type='memory'
            ),
            width=10,
        ),
    ],
    className="mb-3",
)