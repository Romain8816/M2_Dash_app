import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Row import Row
from numpy.core.fromnumeric import size
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
import cchardet as chardet
from detect_delimiter import detect
import dash_daq as daq
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split

from layout.svr_layout import regression_svr
from layout.log_layout import classification_log
from layout.knnclas_layout import classification_KNeighborsClassifier
from layout.knnreg_layout import regression_KNeighborsRegressor
from layout.tree_layout import classification_decision_tree
from layout.lr_layout import Regession_regression_lineaire



# --- /!\ data_path = os.getcwd() +'\data\\' # WINDOWS
data_path = os.getcwd() +'/data/' # LINUX - MAC-OS
files = [f for f in os.listdir(r'%s' %data_path)]

########################################################################################################
# (INIT) Input pour définir le répertoire de travail
location_folder = dbc.Row(
    [
        dbc.Col(
            dbc.Input(
                    type="text", id="location_folder", placeholder="Chemin absolu du répertoire du répertoire de travail : C:\..",persistence=True
                ),className="mb-3"
        ),
        dbc.Col(
            dbc.Button(
                "Valider", id="validation_folder", className="me-2", n_clicks=1
            ),className="mb-3"
        )
    ]
)

########################################################################################################
# (INIT) Dropdown pour sélectionner le fichier à analyser selon le répertoire de travail choisit
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
                persistence = True
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

########################################################################################################
# (INIT) Dropdown pour sélectionner la variable cible
target_selection = dbc.Row(
    [
        dbc.Label("Variable cible", html_for="target_selection", width=1,style={'color': 'red','font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                id='target_selection',
                placeholder="Sélectionner la variable cible",
                searchable=False,clearable=False,
                style={'width':'50%'},
                persistence=True,
                #persistence_type='memory'
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

########################################################################################################
# (INIT) Dropdown pour sélection des variables explicatives
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
                    persistence=True,
                    #persistence_type='memory'
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

########################################################################################################
# (Classification) Onglets

classification_tabs = dbc.Tabs(
    id="classification_tab",
    children= [
        dbc.Tab(classification_log,label="Régression logistique",tab_id ='log',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(classification_decision_tree,label="Arbre de décision",tab_id='decision_tree',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(classification_KNeighborsClassifier,label="KNeighborsClassifier", tab_id='KNeighborsClassifier',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'})
    ]
)
########################################################################################################
# (Régression) Onglets

regression_tabs = dbc.Tabs(
    id='regression_tabs',
    children = [
        dbc.Tab(Regession_regression_lineaire,label="Régression linéaire",tab_id='reg_lin'),
        dbc.Tab(regression_svr,label="SVR",tab_id ='svr'),
        dbc.Tab(regression_KNeighborsRegressor,label="KNeighborsRegressor", tab_id='KNeighborsRegressor',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'})
    ]
)
