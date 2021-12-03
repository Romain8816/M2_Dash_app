import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

######################################
# Le code ci-dessous définit l'affichage
# des options sur la page web pour
# l'algorithme de réression linéaire
######################################
Regession_regression_lineaire = dbc.Card(
    children=[

            html.H2(html.B(html.P("Linear Regression", className="card-text"))),html.Br(),html.Hr(),html.Br(),

            html.Div(
                [
                     html.H4(html.B("Paramètres généraux")),html.Br(),
                     dbc.Row(
                         [
                             dbc.Col(
                                dbc.Label("Taille de l'échantillon test", html_for="lr_test_size",style={'font-weight': 'bold'}),
                                width=4
                            ),
                            dbc.Col(
                                dcc.Input(id="lr_test_size", type="number", placeholder="input with range",min=0.1,max=0.5, step=0.1,value=0.3)
                            )
                        ]
                    ),

                    html.B("Random state "),html.I("par défaut=42"),html.Br(),
                    html.P("Contrôle le brassage appliqué aux données avant d'appliquer le fractionnement. Passer un int pour une sortie reproductible sur plusieurs appels de fonction.", className="card-text"),
                    dcc.Input(id="lr_random_state", type="number", placeholder="input with range",min=1,max=42, step=1,value=42),html.Br(),html.Br(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Centrer réduire",  html_for="lr_centrer_reduire",style={'font-weight': 'bold'}),
                                ], width=1
                            ),
                            dbc.Col(
                                dbc.Checklist(
                                    id="lr_centrer_reduire",
                                    options=[{"value":"yes"}]
                                )
                            )
                        ]
                    ),

                ]
            ),

            html.Div([

                html.Div([


                    html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),html.Br(),
                    html.B("GridSearchCV_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                    dcc.Input(
                        id="Linear_GridSearchCV_number_of_folds",
                        type="number",
                        placeholder="input with range",
                        min=1,max=100, step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("GridSearchCV_scoring "),
                    html.I("par défaut = 'R2'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour l'optimisation des hyperparamètres."),
                    dcc.Dropdown(
                        id='Linear_GridSearchCV_scoring',
                        options=[
                            {'label': "MSE", 'value': "MSE"},
                            {'label': "MAE", 'value': "MAE"},
                            {'label':"RMSE",'value':'RMSE'}

                        ],
                        value = 'MSE'
                    ),html.Br(),html.Br(),
                    html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),
                    html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                    dcc.Dropdown(
                        id="Linear_GridSearchCV_njobs",
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
                    ),html.Br(),html.Br(),
                    dbc.Button("valider GridSearchCV", color="info",id='Linear_button_GridSearchCV',n_clicks=0),
                    dcc.Loading(id="ls-loading-0_linear", children=[html.Div(id="ls-loading-output-0_linear")], type="default"),html.Br(),html.Hr(),html.Br(),
                    # on passe à une autre section (modification des paramètre) >> propre à la fonction
                    html.H4(html.B("Paramètres : ")),
                    html.Hr(),html.Br(),
                    html.B("fit_intercept : ")," ",
                    dcc.Dropdown(
                        id='fit_intercept',
                        options=[
                            {'label': 'True', 'value': 'True'},
                            {'label': 'False', 'value': 'False'}
                        ],
                        value= 'True',
                    ),
                    html.Br(),
                    html.B("copy_X : "),"",
                    dcc.Dropdown(
                        id='copy_X',
                        options=[
                            {'label': 'True', 'value': 'True'},
                            {'label': 'False', 'value': 'False'}],
                        value= 'True',
                    ),
                    html.Br(),
                    html.B("n_jobs : "),"",
                    dcc.Input(
                        id='n_jobs',
                        type='number',
                        min=0,
                        max=10,
                        value=0,
                    ),
                    html.Br(),html.Br(),
                    dbc.Button("Valider Fit & Predict", color="danger",id='Linear_button_FitPredict',n_clicks=0),
                    dcc.Loading(id="ls-loading-1_linear", children=[html.Div(id="ls-loading-output-1_Linear")], type="default"),html.Br(),html.Hr(),html.Br(),
                    # on passe à une autre section (validation croissé)
                    html.H4(html.B("validation croisée :")),html.Br(),html.Br(),
                    html.B("cv_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                    dcc.Input(
                        id="Linear_cv_number_of_folds", type="number",
                        placeholder="input with range",min=1,max=100,
                        step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("cv_scoring "),html.I("par défaut = 'R2'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                    dcc.Dropdown(
                        id='Linear_cv_scoring',
                        options=[
                            {'label': "MSE", 'value': "MSE"},
                            {'label': "MAE", 'value': "MAE"},
                            {'label':"RMSE",'value':'RMSE'}

                        ],
                        value = 'MSE'
                    ),html.Br(),html.Hr(),
                    dbc.Button("Valider CrossValidation", color="success",id='Linear_button_CrossValidation',n_clicks=0),
                    dcc.Loading(id="ls-loading-2_Linear", children=[html.Div(id="ls-loading-output-2_Linear")], type="default")

                ],className="six columns",style={"margin":10}),

                html.Div([

                    html.H3(html.B("Résultats :")),
                    html.Div(id="res_Linear_GridSearchCV"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_Linear_FitPredict"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_Linear_CrossValidation"), html.Hr(),

                ],className="six columns",style={"margin":10}),

            ], className="row"),
        ],body=True
)
