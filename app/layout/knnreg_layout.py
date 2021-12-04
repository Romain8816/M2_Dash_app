import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

######################################
# Le code ci-dessous définit l'affichage
# des options sur la page web pour
# de l'algorithme de regression KNN
######################################
regression_KNeighborsRegressor = dbc.Card(
    children=[

        html.H2(html.B(html.P("KNeighbors Regressor", className="card-text"))),
        html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),

        html.Div(
            [
                 html.H4(html.B("Paramètres généraux")),html.Br(),
                 dbc.Row(
                     [
                         dbc.Col(
                            [
                            dbc.Label("Taille de l'échantillon test", html_for="KNeighborsRegressor_test_size",style={'font-weight': 'bold'}),
                             ],width=3
                        ),
                        dbc.Col(
                            [
                            dcc.Slider(id='KNeighborsRegressor_test_size',min=0.1,max=0.5,step=0.1,value=0.3,tooltip={"placement": "bottom", "always_visible": True}),
                            ],width=2
                        ),
                        dbc.Col(
                           width=2
                       ),
                        dbc.Col(
                           [
                           html.B("Random state "),html.I("par défaut=42"),html.P(" Contrôle le brassage appliqué aux données avant d'appliquer le fractionnement. Passer un int pour une sortie reproductible sur plusieurs appels de fonction.", className="card-text"),
                           ],width=3
                       ),
                       dbc.Col(
                          [
                          dcc.Input(id="KNeighborsRegressor_random_state", type="number", placeholder="input with range",min=1,max=42, step=1,value=42),html.Br(),html.Br(),
                          ],width=1
                       )
                    ]
                ),
               dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Centrer réduire",  html_for="KNeighborsRegressor_centrer_reduire",style={'font-weight': 'bold'}),
                            ], width=3
                        ),
                        dbc.Col(
                            [
                            dbc.Checklist(
                                id="KNeighborsRegressor_centrer_reduire",
                                options=[{"value":"yes"}]
                            )
                            ],width=1
                        )
                    ]
                ),
                html.Br(),
                dbc.Row(
                     [
                         dbc.Col(
                             [
                                html.B("shuffle "),html.I("par défaut shuffle=True"),html.Br(),html.P("s'il faut ou non mélanger les données avant de les diviser.", className="card-text"),
                             ], width=3
                         ),
                         dbc.Col(
                            [
                                dcc.Dropdown(
                                    id='KNeighborsRegressor_shuffle',
                                    options=[
                                        {'label': 'True', 'value': 'True'},
                                        {'label': 'False', 'value': 'False'},
                                    ],
                                    value = 'True'
                                )
                                ], width=1
                             ),
                          dbc.Col(
                               width=2
                              ),
                          ]
                         ),
                html.Br(),
            ]
        ),

        html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),

        html.Div([

            html.Div([

                html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),

                html.B("GridSearchCV_number_of_folds "),html.I("par défaut=5"),html.Br(),html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_GridSearchCV_number_of_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),

                html.B("GridSearchCV_scoring "),html.I("par défaut = 'MSE'"),html.Br(),html.P("Selectionnez la méthode de scoring pour l'optimisation des hyperparamètres."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_GridSearchCV_scoring',
                    options=[
                        {'label': "MSE", 'value': "MSE"},
                        {'label': "RMSE", 'value': "RMSE"},
                        {'label': "MAE", 'value': "MAE"},
                    ],
                    value = 'MSE'
                ),html.Br(),html.Br(),

                html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                dcc.Dropdown(
                    id="KNeighborsRegressor_GridSearchCV_njobs",
                    options=[
                        {'label': -1, 'value': -1},{'label': 1, 'value': 1},{'label': 2, 'value': 2},{'label': 3, 'value': 3},{'label': 4, 'value': 4},
                        {'label': 5, 'value': 5},{'label': 6, 'value': 6},{'label': 7, 'value': 7},{'label': 8, 'value': 8},{'label': 9, 'value': 9},
                        {'label': 10, 'value': 10},{'label': 11, 'value': 11},{'label': 12, 'value': 12},{'label': 13, 'value': 13},{'label': 14, 'value': 14},
                        {'label': 15, 'value': 15},{'label': 16, 'value': 16},{'label': 17, 'value': 17},{'label': 18, 'value': 18},{'label': 19, 'value': 19},
                        {'label': 20, 'value': 20},{'label': 21, 'value': 21},{'label': 22, 'value': 22},{'label': 23, 'value': 23},{'label': 24, 'value': 24},
                        {'label': 25, 'value': 25},{'label': 26, 'value': 26},{'label': 27, 'value': 27},{'label': 28, 'value': 28},{'label': 29, 'value': 29},
                        {'label': 30, 'value': 30},{'label': 31, 'value': 31},{'label': 32, 'value': 32},{'label': 'None', 'value': 'None'}
                    ],
                    value = -1
                ),html.Br(),

                dbc.Button("valider GridSearchCV", color="info",id='KNeighborsRegressor_button_GridSearchCV',n_clicks=0),
                dcc.Loading(
                    id="KNeighborsRegressor-ls-loading-1",
                    children=[html.Div(id="KNeighborsRegressor-ls-loading-output-1")], type="default"
                ),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),

                html.H4(html.B("Performance du modèle sur le jeu de test :")),html.Br(),html.Br(),
                html.B("n_neighbors "),html.I("par défaut=5"),html.Br(),html.P("Nombre de voisins à utiliser par défaut pour les requêtes de voisins.", className="card-text"),

                dcc.Input(id="KNeighborsRegressor_n_neighbors", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),
                html.B("weights "),html.I("par défaut = 'uniform'"),html.Br(),html.P("Fonction de poids utilisée dans la prédiction."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_weights',
                    options=[
                        {'label': 'uniform', 'value': 'uniform'},
                        {'label': 'distance', 'value': 'distance'},
                    ],
                    value = 'uniform'
                ),html.Br(),html.Br(),

                html.B("algorithm "),html.I("par défaut = 'auto'"),html.Br(),html.P("Algorithme utilisé pour calculer les voisins les plus proches."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_algorithm',
                    options=[
                        {'label': 'auto', 'value': 'auto'},
                        {'label': 'ball_tree', 'value': 'ball_tree'},
                        {'label': 'kd_tree', 'value': 'kd_tree'},
                        {'label': 'brute', 'value': 'brute'},
                    ],
                    value = 'auto'
                ),html.Br(),html.Br(),
                html.B("leaf_size "),html.I("par défaut=30"),html.Br(),html.P("Taille de la feuille transmise à BallTree ou KDTree. Cela peut affecter la vitesse de construction et de requête, ainsi que la mémoire requise pour stocker l'arbre. La valeur optimale dépend de la nature du problème.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_leaf_size", type="number", placeholder="input with range",min=1,max=300, step=1,value=30),html.Br(),html.Br(),
                html.B("p "),html.I("par défaut=2"),html.Br(),html.P("Paramètre de puissance pour la métrique de Minkowski. Lorsque p = 1, cela équivaut à utiliser manhattan_distance (l1) et euclidean_distance (l2) pour p = 2. Pour p arbitraire, minkowski_distance (l_p) est utilisé.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_p", type="number", placeholder="input with range",min=1,max=100, step=1,value=2),html.Br(),html.Br(),
                html.B("metric "),html.I("par défaut = 'minkowski'"),html.Br(),html.P("La métrique de distance à utiliser pour l'arbre."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_metric',
                    options=[
                        {'label': 'minkowski', 'value': 'minkowski'},
                        {'label': 'euclidean', 'value': 'euclidean'},
                        {'label': 'manhattan', 'value': 'manhattan'},
                        {'label': 'chebyshev', 'value': 'chebyshev'},
                        {'label': 'wminkowski', 'value': 'wminkowski'},
                        {'label': 'seuclidean', 'value': 'seuclidean'}
                    ],
                    value = 'minkowski'
                ),html.Br(),

                dbc.Button("Valider Fit & Predict", color="danger",id='KNeighborsRegressor_button_FitPredict',n_clicks=0),
                dcc.Loading(id="KNeighborsRegressor-ls-loading-3", children=[html.Div(id="KNeighborsRegressor-ls-loading-output-3")], type="default"),html.Br(),
                html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),
                html.H4(html.B("Validation croisée :")),html.Br(),html.Br(),
                html.B("cv_number_of_folds "),html.I("par défaut=5"),html.Br(),html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_cv_number_of_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),
                html.B("cv_scoring "),html.I("par défaut = 'MSE'"),html.Br(),html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_cv_scoring',
                    options=[
                        {'label': "MSE", 'value': "MSE"},
                        {'label': "RMSE", 'value': "RMSE"},
                        {'label': "MAE", 'value': "MAE"},
                    ],
                    value = 'MSE'
                ),html.Br(),
                dbc.Button("Valider K-Fold Cross-Validation", color="success",id='KNeighborsRegressor_button_CrossValidation',n_clicks=0),
                dcc.Loading(id="KNeighborsRegressor-ls-loading-2", children=[html.Div(id="KNeighborsRegressor-ls-loading-output-2")], type="default")


                ],className="six columns",style={"margin":10}),

            html.Div([

                    html.H4(html.B("Résultats :")),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),html.Br(),
                    html.Div(id="res_KNeighborsRegressor_GridSearchCV"),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),html.Br(),
                    html.Div(id="res_KNeighborsRegressor_FitPredict"),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),html.Br(),
                    html.Div(id="res_KNeighborsRegressor_CrossValidation")

                ],className="six columns",style={"margin":10})

        ], className="row")

    ],
    body=True
)
