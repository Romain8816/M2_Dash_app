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
import cchardet as chardet
from detect_delimiter import detect
from sklearn.cluster import KMeans
import dash_daq as daq
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split

# --- /!\ data_path = os.getcwd() +'\data\\' # WINDOWS
data_path = os.getcwd() +'/data/' # LINUX - MAC-OS
files = [f for f in os.listdir(r'%s' %data_path)]

########################################################################################################
# (INIT) Input pour définir le répertoire de travail
location_folder = dbc.Row(
    [
        dbc.Col(
            dbc.Input(
                    type="text", id="location_folder", placeholder="Chemin absolu du répertoire du répertoire de travail : C:\.."
                ),className="mb-3"
        ),
        dbc.Col(
            dbc.Button(
                "Valider", id="validation_folder", className="me-2", n_clicks=0
            ),className="mb-3"
        )
    ]
)
########################################################################################################
# Composant qui permet de déposer un ficher (le garder ou pas ?)
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
                persistence =False
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
                persistence=False,
                persistence_type='memory'
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
                    persistence=False,
                    persistence_type='memory'
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

########################################################################################################
# (Classification) Onglets

# Arbre de décision
classification_decision_tree = dbc.Card(
    children = [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(html.B("Paramètres pour la méthode des arbres de décison")),
                            html.Hr(),html.Br(),
                            html.B("criterion : ")," ",
                            dcc.Dropdown(
                                id='criterion',
                                options=[
                                    {'label': 'gini', 'value': 'gini'},
                                    {'label': 'entropy', 'value': 'entropy'}
                                ],
                                value='gini'
                            ),
                            html.Br(),
                            html.B("splitter : "),"",
                            dcc.Dropdown(
                                id='splitter',
                                options=[
                                    {'label':'best','value':'best'},
                                    {'label':'random','value':'random'}],
                                value="best",
                            ),
                            html.Br(),
                            html.B("max_depth : "),"",
                            dcc.Input(
                                id='max_depth',
                                type='number',
                                min=0,
                                max=10,
                                value=0,
                            ),
                            html.Br(),
                            html.B("min_samples_split : "),"",
                            html.Br(),
                            dcc.Input(
                                id='min_samples_split',
                                type="number",
                                value=2,
                                max=10,
                                min=1
                            ),
                            html.Br(),
                            html.B("min_samples_leaf : "),"",
                            dcc.Input(
                                id='min_samples_leaf',
                                type='number',
                                min=1,
                                max=10,
                                value=1
                            ),
                            html.Br(),
                            html.B("max_leaf_nodes : "),"",
                            dcc.Input(
                                id='max_leaf_nodes',
                                type='number',
                                max=10,
                                min=0,
                                value=0
                            ),
                            html.Br(),
                            dbc.Button("Valider", color="danger",id='tree_button',n_clicks=0)
                        ],className="six columns",

                    ),
                    html.Br(),
                    html.Br(),
                    html.Div
                        (
                            [
                                html.H4(html.B("Exploration des résultats")),
                                html.Hr(),html.Br(),
                                dcc.Dropdown(
                                    id="diff_metric",
                                    options=[
                                        {"label":"accuracy","value":"accuracy"},
                                        {"label":"f1 score","value":"f1_macro"},
                                        {"label":"recall","value":"recall_macro"},
                                        {"label":"precision","value":"precision_macro"}],
                                    value="accuracy"),
                                html.Br(),
                                html.Div(id='print_result_metric'),
                                html.Br(),
                                html.H4("Visualisation de l'arbre de decsion : "),
                                dbc.Button("Viszualise tree", color="danger",id='plot_button',n_clicks=0)
                                ], className="six columns"
                        ),
                ],
            ),
        ],
    #body=True
)

# SVM
classification_SVM = dbc.Card(
    children=[
        html.P("Support Vector Machine", className="card-text"),
        dbc.Label("Taille de l'échantillion de test (compris entre 0.0 et 1.0)", html_for="test_size", width=4,style={'font-weight': 'bold'}),
        dcc.Slider(
            id='test_size',
            min=0.0,
            max=1.0,
            step=0.1,
            value=0.3,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        dbc.Label("Random seed", html_for="random_state", width=4,style={'font-weight': 'bold'}),
        dbc.Input(id='random_state',type='number'),
        dbc.Label("Nombre de fold pour la validation croisée", html_for="k_fold", width=4,style={'font-weight': 'bold'}),
        dbc.Input(id='k_fold',value=5,type='number'),
        dbc.Label("Type de noyau (kernel)", html_for="kernel_selection", width=4,style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='kernel_selection',
            options=[
                {'label': 'linéaire', 'value': 'linear'},
                {'label': 'polynomial', 'value': 'poly'},
                {'label': 'RBF', 'value': 'rbf'},
                {'label': 'Sigmoïde', 'value': 'sigmoid'},
            ],
            value = 'rbf'
        ),
        dbc.Label("Paramètre de régularisation (C)", html_for="regularisation_selection", width=4,style={'font-weight': 'bold'}),
        dcc.Slider(
            id='regularisation_selection',
            min=0.1,
            max=100,
            step=0.5,
            value=0.1,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
         dbc.Button("Valider", color="danger",id='smv_button',n_clicks=0),
         html.Div(id='res_svm')
    ],
    body=True
)

# KNN
classification_KNN = dbc.Card(
    dbc.CardBody(
        [
            html.P("KNN", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)

# CAH
classification_CAH = dbc.Card(
    dbc.CardBody(
        [
            html.P("CAH", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)

# KMeans
classification_Kmeans = dbc.Card(
    children=[
        html.Div
        (
            [
                html.Div
                (
                    [
                        html.H4(html.B("Paramètres pour la méthode Kmeans")),
                        html.Hr(),html.Br(),
                        html.B("n_clusters : ")," défault nombre de modalités de la variable cible, Le nombre de clusters à former ainsi que le nombre de centroïdes à générer.",
                        dcc.Slider(
                            id='n_clusters',
                            min=0,
                            max=10,
                            step=1,
                            value=5,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        html.B("init : "),"default=k-means++, Méthode d'initialisation",
                        dcc.Dropdown(
                            id='init',
                            options=[
                                {'label': 'k-means++', 'value': 'k-means++'},{'label': 'random', 'value': 'random'}
                            ],
                            value='k-means++'
                        ),
                        html.Br(),
                        html.B("n_init : "),"default=10, Nombre de fois où l'algorithme k-means sera exécuté avec différentes graines centroïdes. Les résultats finaux seront le meilleur résultat de n_init exécutions consécutives en termes d'inertie.",
                        daq.NumericInput(
                            id='n_init',
                            min=1,
                            max=100,
                            value=10,
                        ),
                        html.Br(),
                        html.B("max_iter : "),"default=300, Nombre maximum d'itérations de l'algorithme des k-moyennes pour une seule exécution.",
                        daq.NumericInput(
                            id='max_iter',
                            min=1,
                            max=1000,
                            value=300,
                        ),
                        html.Br(),
                        html.B("tol : "),"default=1e-4, Tolérance relative par rapport à la norme de Frobenius de la différence des centres de cluster de deux itérations consécutives pour déclarer la convergence.",html.Br(),
                        dcc.Input(
                            id='tol',
                            type="number",
                            value=0.0001,
                            max=0.1,
                            min=0.0000000001
                        ),
                        html.Br(),html.Br(),
                        html.B("verbose : "),"default=0, mode verbosité",
                        daq.NumericInput(
                            id='verbose',
                            min=0,
                            max=1,
                            value=0
                        ),
                        html.Br(),
                        html.B("random_state : "),"defaut=None, Détermine la génération de nombres aléatoires pour l'initialisation du centroïde. Utilisez un int pour rendre le caractère aléatoire déterministe.",
                        dcc.Dropdown(
                            id='random_state',
                            options=[
                                {'label': 'None', 'value': 'None'},{'label': '0', 'value': '0'},{'label': '1', 'value': '1'},{'label': '2', 'value': '2'},{'label': '3', 'value': '3'},{'label': '4', 'value': '4'},{'label': '5', 'value': '5'},{'label': '6', 'value': '6'},{'label': '7', 'value': '7'},{'label': '8', 'value': '8'},{'label': '9', 'value': '9'},{'label': '10', 'value': '10'},{'label': '11', 'value': '11'},{'label': '12', 'value': '12'},{'label': '13', 'value': '13'},{'label': '14', 'value': '14'},{'label': '15', 'value': '15'},{'label': '16', 'value': '16'},{'label': '17', 'value': '17'},{'label': '18', 'value': '18'},{'label': '19', 'value': '19'},{'label': '20', 'value': '20'},{'label': '21', 'value': '21'},{'label': '22', 'value': '22'},{'label': '23', 'value': '23'},{'label': '24', 'value': '24'},{'label': '25', 'value': '25'},{'label': '26', 'value': '26'},{'label': '27', 'value': '27'},{'label': '28', 'value': '28'},{'label': '29', 'value': '29'},{'label': '30', 'value': '30'},{'label': '32', 'value': '31'},{'label': '32', 'value': '32'},{'label': '33', 'value': '33'},{'label': '34', 'value': '34'},{'label': '35', 'value': '35'},{'label': '36', 'value': '36'},{'label': '37', 'value': '37'},{'label': '38', 'value': '38'},{'label': '39', 'value': '39'},{'label': '40', 'value': '40'},{'label': '41', 'value': '41'},{'label': '42', 'value': '42'}
                            ],
                            value='None'
                        ),
                        html.Br(),
                        html.B("algorithm : "),"default='auto', Algorithme K-means à utiliser. L'algorithme classique de style EM est « complet ». La variation « elkan » est plus efficace sur des données avec des clusters bien définis, en utilisant l'inégalité triangulaire. Cependant, il est plus gourmand en mémoire en raison de l'allocation d'un tableau supplémentaire de formes (n_samples, n_clusters).",
                        dcc.Dropdown(
                            id='algorithm',
                            options=[
                                {'label': 'auto', 'value': 'auto'},{'label': 'full', 'value': 'full'},{'label': 'elkan', 'value': 'elkan'}
                            ],
                            value='auto'
                        )
                    ],className="six columns"
                ),
                    html.Div
                    (
                        [
                            html.H4(html.B("Exploration des résultats")),
                            html.Hr(),html.Br(),
                            dcc.Dropdown(id="kmeans-explore-object"),
                            html.Br(),
                            html.Div(id='kmeans-explore-object-display'),
                            html.Br(),
                            dcc.Graph(id='kmeans-pca',style={'width': '90vh', 'height': '90vh'}),
                            html.Br(),
                            dcc.Graph(id='input-pca',style={'width': '90vh', 'height': '90vh'})
                        ], className="six columns"
                    ),
            ],id="kmeans-container",style = {"margin":25,"display":"none"}, className="row"
        )
    ],className="mt-3",
)

# Ensemble des onglets de classification
classification_tabs = dbc.Tabs(
    id="classification_tab",
    children= [
        dbc.Tab(classification_decision_tree,label="Arbre de décision",tab_id='decision_tree',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(classification_SVM,label="SVM",tab_id ='svm',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(classification_KNN,label="KNN",tab_id ='knn',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(classification_CAH,label="CAH",tab_id ='cah',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(label ="Kmeans", tab_id='kmeans',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'})
    ]
)

########################################################################################################
# (Régression) Onglets

regression_tabs = dbc.Tabs(
    id='regression_tabs',
    children = [
        dbc.Tab(label="Régression linéaire",tab_id='reg_lin'),
        dbc.Tab(label="Régression polynomiale",tab_id ='reg_poly'),
        dbc.Tab(label="Régression lasso",tab_id ='reg_lasso')
    ]
)


# Onglets pour les algos de régression -------------------------------------------------------------------------------------------------------


kmeans_params_and_results = html.Div([
    html.Div([
            html.H4(html.B("Paramètres pour la méthode Kmeans")),
            html.Hr(),html.Br(),
            html.B("n_clusters : ")," défault nombre de modalités de la variable cible, Le nombre de clusters à former ainsi que le nombre de centroïdes à générer.",
            dcc.Slider(
                id='n_clusters',
                min=0,
                max=10,
                step=1,
                value=3,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            html.B("init : "),"default=k-means++, Méthode d'initialisation",
            dcc.Dropdown(
                id='init',
                options=[
                    {'label': 'k-means++', 'value': 'k-means++'},{'label': 'random', 'value': 'random'}
                ],
                value='k-means++'
            ),
            html.Br(),
            html.B("n_init : "),"default=10, Nombre de fois où l'algorithme k-means sera exécuté avec différentes graines centroïdes. Les résultats finaux seront le meilleur résultat de n_init exécutions consécutives en termes d'inertie.",
            daq.NumericInput(
                id='n_init',
                min=1,
                max=100,
                value=10,
            ),
            html.Br(),
            html.B("max_iter : "),"default=300, Nombre maximum d'itérations de l'algorithme des k-moyennes pour une seule exécution.",
            daq.NumericInput(
                id='max_iter',
                min=1,
                max=1000,
                value=300,
            ),
            html.Br(),
            html.B("tol : "),"default=1e-4, Tolérance relative par rapport à la norme de Frobenius de la différence des centres de cluster de deux itérations consécutives pour déclarer la convergence.",html.Br(),
            dcc.Input(
                id='tol',
                type="number",
                value=0.0001,
                max=0.1,
                min=0.0000000001
            ),
            html.Br(),html.Br(),
            html.B("verbose : "),"default=0, mode verbosité",
            daq.NumericInput(
                id='verbose',
                min=0,
                max=1,
                value=0
            ),
            html.Br(),
            html.B("random_state : "),"defaut=None, Détermine la génération de nombres aléatoires pour l'initialisation du centroïde. Utilisez un int pour rendre le caractère aléatoire déterministe.",
            dcc.Dropdown(
                id='random_state',
                options=[
                    {'label': 'None', 'value': 'None'},{'label': '0', 'value': '0'},{'label': '1', 'value': '1'},{'label': '2', 'value': '2'},{'label': '3', 'value': '3'},{'label': '4', 'value': '4'},{'label': '5', 'value': '5'},{'label': '6', 'value': '6'},{'label': '7', 'value': '7'},{'label': '8', 'value': '8'},{'label': '9', 'value': '9'},{'label': '10', 'value': '10'},{'label': '11', 'value': '11'},{'label': '12', 'value': '12'},{'label': '13', 'value': '13'},{'label': '14', 'value': '14'},{'label': '15', 'value': '15'},{'label': '16', 'value': '16'},{'label': '17', 'value': '17'},{'label': '18', 'value': '18'},{'label': '19', 'value': '19'},{'label': '20', 'value': '20'},{'label': '21', 'value': '21'},{'label': '22', 'value': '22'},{'label': '23', 'value': '23'},{'label': '24', 'value': '24'},{'label': '25', 'value': '25'},{'label': '26', 'value': '26'},{'label': '27', 'value': '27'},{'label': '28', 'value': '28'},{'label': '29', 'value': '29'},{'label': '30', 'value': '30'},{'label': '32', 'value': '31'},{'label': '32', 'value': '32'},{'label': '33', 'value': '33'},{'label': '34', 'value': '34'},{'label': '35', 'value': '35'},{'label': '36', 'value': '36'},{'label': '37', 'value': '37'},{'label': '38', 'value': '38'},{'label': '39', 'value': '39'},{'label': '40', 'value': '40'},{'label': '41', 'value': '41'},{'label': '42', 'value': '42'}
                ],
                value='None'
            ),
            html.Br(),
            html.B("algorithm : "),"default='auto', Algorithme K-means à utiliser. L'algorithme classique de style EM est « complet ». La variation « elkan » est plus efficace sur des données avec des clusters bien définis, en utilisant l'inégalité triangulaire. Cependant, il est plus gourmand en mémoire en raison de l'allocation d'un tableau supplémentaire de formes (n_samples, n_clusters).",
            dcc.Dropdown(
                id='algorithm',
                options=[
                    {'label': 'auto', 'value': 'auto'},{'label': 'full', 'value': 'full'},{'label': 'elkan', 'value': 'elkan'}
                ],
                value='auto'
            )
        ], className="six columns"),
        html.Div([
            html.H4(html.B("Exploration des résultats")),
            html.Hr(),html.Br(),
            dcc.Dropdown(
                id="kmeans-explore-object"
            ),
            html.Br(),
            html.Div(id='kmeans-explore-object-display'),
            html.Br(),
            dcc.Graph(
                id='kmeans-pca',style={'width': '90vh', 'height': '90vh'}
            ),
            html.Br(),
            dcc.Graph(
                id='input-pca',style={'width': '90vh', 'height': '90vh'}
            )
        ], className="six columns"),
    ],id="kmeans-container",style = {"margin":25,"display":"none"}, className="row"
)
