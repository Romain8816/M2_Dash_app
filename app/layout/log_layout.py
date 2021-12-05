from dash import dcc
from dash import html
import dash_bootstrap_components as dbc



classification_log = dbc.Card(          
    children=[
        html.H2(html.B(html.P("Logistic Regression", className="card-text"))),
        html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),
        html.Div(
                [
                     html.H4(html.B("Paramètres généraux")),html.Br(),
                     dbc.Row(
                         [
                             dbc.Col(
                                [
                                dbc.Label("Taille de l'échantillon test", html_for="log_test_size",style={'font-weight': 'bold'}),
                                 ],width=3
                            ),
                            dbc.Col(
                                [
                                dcc.Slider(id='log_test_size',min=0.1,max=0.5,step=0.1,value=0.3,tooltip={"placement": "bottom", "always_visible": True}),
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
                              dcc.Input(id="log_random_state", type="number", placeholder="input with range",min=1,max=42, step=1,value=42),html.Br(),html.Br(),
                              ],width=1
                           )
                        ]
                    ),
                   dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Centrer réduire",  html_for="log_centrer_reduire",style={'font-weight': 'bold'}),
                                ], width=3
                            ),
                            dbc.Col(
                                [
                                dbc.Checklist(
                                    id="log_centrer_reduire",
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
                                        id='log_shuffle',
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
                                dbc.Col(
                                    [
                                       html.B("stratify "),html.I("par defaut stratify=False"),html.Br(),
                                       html.P("Si ce n'est pas False, les données sont divisées de manière stratifiée en utilisant les étiquettes de la classe à prédire", className="card-text"),
                                    ], width=3
                                ),
                                dbc.Col(
                                   [
                                       dcc.Dropdown(
                                           id='log_stratify',
                                           options=[
                                               {'label': 'True', 'value': 'True'},
                                               {'label': 'False', 'value': 'False'},
                                           ],
                                           value = 'False'
                                       )
                                       ], width=1
                                    )
                              ]
                             ),
                    html.Br(),
                ]
            ),

            html.Br(),
            html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),

        html.Div(
            [
                html.Div(
                    children = 
                    [
                        html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),

                        html.B("GridSearchCV_number_of_folds "),html.I("par défaut=5"),html.Br(),

                        html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                        dcc.Input(id="log_gridCV_k_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),
                    
                        html.B("GridSearchCV_scoring "),html.I("par défaut = 'f1_macro'"),html.Br(),
                        html.P("Selectionner la méthode de scoring pour l'optimisation des hyperparamètres."),

                        dcc.Dropdown(
                            id='log_gridCV_scoring',
                            options=[
                                {'label': "accuracy", 'value': "accuracy"},
                                {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                                {'label': "f1_binary", 'value': "f1_binary"},
                                {'label': "f1_micro", 'value': "f1_micro"},
                                {'label': "f1_macro", 'value': "f1_macro"},
                                {'label': "f1_weighted", 'value': "f1_weighted"},
                                {'label': "precision_binary", 'value': "precision_binary"},
                                {'label': "precision_micro", 'value': "precision_micro"},
                                {'label': "precision_macro", 'value': "precision_macro"},
                                {'label': "precision_weighted", 'value': "precision_weighted"},
                                {'label': "recall_binary", 'value': "recall_binary"},
                                {'label': "recall_micro", 'value': "recall_micro"},
                                {'label': "recall_macro", 'value': "recall_macro"},
                                {'label': "recall_weighted", 'value': "recall_weighted"},
                                {'label': "roc_auc_ovr", 'value': "roc_auc_ovr"},
                                {'label': "roc_auc_ovo", 'value': "roc_auc_ovo"},
                                {'label': "roc_auc_ovr_weighted", 'value':"roc_auc_ovr_weighted" },
                                {'label': "roc_auc_ovo_weighted", 'value': "roc_auc_ovo_weighted"}
                            ],
                            value = 'f1_macro'
                        ),html.Br(),html.Br(),

                        html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),
                        html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                        dcc.Dropdown(
                            id="log_GridSearchCV_njobs",
                            options= [{'label': 'None', 'value': 'None'}] + [{'label': -1, 'value': -1}] + [{'label':i, 'value':i} for i in range(1,33)],
                            value = -1
                        ),html.Br(),html.Br(),

                        dbc.Button("valider GridSearchCV",color ="info",id='log_button_GridSearchCV',n_clicks=0),
                        
                        html.Br(),html.Hr(),
                        
                        html.H4(html.B("Paramètrage du modèle et Fit & Predict :")),html.Br(),
                        
                        # Paramètres de l'algo
                        dbc.Row(
                            [
                                # Pénalité
                                dbc.Col(
                                    [
                                        dbc.Label("Pénalité", html_for="log_penalty",style={'font-weight': 'bold'}),
                                        dcc.Dropdown(
                                            id='log_penalty',
                                            options=[
                                                {'label': 'aucune', 'value': 'none'},
                                                {'label': 'l1', 'value': 'l1'},
                                                {'label': 'l2', 'value': 'l2'},
                                                {'label': 'elasticnet', 'value': 'elasticnet'},
                                            ],
                                            value = 'none'
                                        ),
                                    ],
                                ),

                                # l1 ratio
                                dbc.Col(
                                    [
                                        dbc.Label("l1 ratio", html_for="log_l1_ratio",style={'font-weight': 'bold'}),
                                        dcc.Slider(id='log_l1_ratio',min=0,max=1,step=0.1,value=0,tooltip={"placement": "bottom", "always_visible": True}),
                                    ],
                                )
                            ]
                        ),

                        html.Br(),
                        dbc.Row(
                            [
                                # Paramètre de régularisation
                                dbc.Col(
                                    [
                                        dbc.Label("Régularisation (C)", html_for="log_regularisation",style={'font-weight': 'bold'}),
                                        dbc.Input(id='log_regularisation',type='number',min=0,max=5,step=0.1,value=0.1,),
                                    ],
                                ),
                            ],style={'margin-bottom': '1em'}
                        ),

                        dbc.Row(
                            [
                                # Solver 
                                dbc.Col(
                                    [
                                        dbc.Label("Solver",html_for='log_solver',style={'font-weight': 'bold'}),
                                        dcc.Dropdown(
                                            id='log_solver',
                                            options=[
                                                {'label': 'lbfgs', 'value': 'lbfgs'},
                                                {'label': 'newton-cg', 'value': 'newton-cg'},
                                                {'label': 'liblinear', 'value': 'liblinear'},
                                                {'label': 'sag', 'value': 'saga'},
                                            ],
                                            value = 'lbfgs'
                                        )
                                    ],
                                )
                            ]
                        ),
                        html.Br(),
                        dbc.Button("Valider fit & predict", color="danger",id='log_button',n_clicks=0),

                        html.Br(),html.Hr(),


                        # Validation Croisée

                        html.H4(html.B("Validation croisée :")),html.Br(),

                        html.B("cv_number_of_folds "),html.I("par défaut=5"),html.Br(),
                        html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                        dcc.Input(id="log_cv_number_of_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),

                        html.B("cv_scoring "),html.I("par défaut = 'f1_macro'"),html.Br(),
                        html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                        dcc.Dropdown(
                        id='log_cv_scoring',
                        options=[
                            {'label': "accuracy", 'value': "accuracy"},
                            {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                            {'label': "f1_binary", 'value': "f1_binary"},
                            {'label': "f1_micro", 'value': "f1_micro"},
                            {'label': "f1_macro", 'value': "f1_macro"},
                            {'label': "f1_weighted", 'value': "f1_weighted"},
                            {'label': "precision_binary", 'value': "precision_binary"},
                            {'label': "precision_micro", 'value': "precision_micro"},
                            {'label': "precision_macro", 'value': "precision_macro"},
                            {'label': "precision_weighted", 'value': "precision_weighted"},
                            {'label': "recall_binary", 'value': "recall_binary"},
                            {'label': "recall_micro", 'value': "recall_micro"},
                            {'label': "recall_macro", 'value': "recall_macro"},
                            {'label': "recall_weighted", 'value': "recall_weighted"},
                            {'label': "roc_auc_ovr", 'value': "roc_auc_ovr"},
                            {'label': "roc_auc_ovo", 'value': "roc_auc_ovo"},
                            {'label': "roc_auc_ovr_weighted", 'value':"roc_auc_ovr_weighted" },
                            {'label': "roc_auc_ovo_weighted", 'value': "roc_auc_ovo_weighted"}
                        ],
                        value = 'f1_macro'
                    ),html.Br(),

                        dbc.Button("Valider K-Fold Cross-Validation",id='log_button_CrossValidation', color="success", n_clicks=0),                        
                    ],className='col-6'
                ),

                # Div des résultats sur la droite
                html.Div(
                    [
                        html.H3(html.B("Résultats :")),
                        dcc.Loading(
                            children=[html.Div(id="res_log_GridSearchCV")], 
                            type="default"
                        ),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),

                        dcc.Loading(
                            children=[html.Div(id="res_log_FitPredict")], 
                            type="default"
                        ),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),

                        dcc.Loading(
                            children=[html.Div(id="res_log_CrossValidation")], 
                            type="default"
                        ),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"})
                    ],
                    className='col-6'
                )
            ],className="row"
        ),
    ],
    body=True
)