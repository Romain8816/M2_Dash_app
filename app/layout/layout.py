import dash
from dash import dcc
from dash import html
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

data_path = os.getcwd() +'\data\\'
files = [f for f in os.listdir(r'%s' %data_path)]

# Fonction qui permet de lire un fichier csv ou xls et qui retoune un tableau 
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'Il y a eu une erreur dans le format du fichier.'
        ])

    return dbc.Col(
        html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            fixed_rows={'headers': True},
            page_size=20,
            style_cell={'textAlign': 'left','minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
            style_table={'height': '400px', 'overflowY': 'scroll','overflowX': 'scroll'},
            style_header={'backgroundColor': 'dark','fontWeight': 'bold'}
        ),
        html.Hr(),  # horizontal line
    ],className='container-fluid'),
    width=10
)

# Input pour définir le répertoire dans lequel on va choisir le fichier à analyser. 
location_folder = dbc.Row(
    [
        dbc.Col(
            dbc.Input(
                    autocomplete="off",type="text", id="location_folder", placeholder="Veuillez définir le chemin absolu du répertoire dans lequel vous souhaitez travailler au format : C:\.."
                ),className="mb-3"
        ),
        dbc.Col(
            dbc.Button(
                "Valider", id="validation_folder", className="me-2", n_clicks=0
            ),className="mb-3"
        )
    ]
)


# Composant qui permet de déposer un ficher
drag_and_drop = dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=True
)



dataset_selection = dbc.Row(
    [
        dbc.Label("Jeu de données sélectionné", html_for="file_selection", width=1,style={'font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                id='file_selection',
                #options=[{'label':i, 'value':i} for i in files],
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

# Sélection de la variable cible
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


# Sélection des variables explicatives
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