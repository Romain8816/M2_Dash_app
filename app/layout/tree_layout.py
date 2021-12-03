import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

######################################
# Le code ci-dessous définit l'affichage
# des options sur la page web pour
# l'algorithme arbre de décision
######################################
classification_decision_tree = dbc.Card(
    children=[

            html.H2(html.B(html.P("Decision Tree", className="card-text"))),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),


            html.Div(
                [
                     html.H4(html.B("Paramètres généraux")),html.Br(),
                     dbc.Row(
                         [
                             dbc.Col(
                                [
                                dbc.Label("Taille de l'échantillon test", html_for="tree_test_size",style={'font-weight': 'bold'}),
                                 ],width=3
                            ),
                            dbc.Col(
                                [
                                dcc.Slider(id='tree_test_size',min=0.1,max=0.5,step=0.1,value=0.3,tooltip={"placement": "bottom", "always_visible": True}),
                                ],width=1
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
                              dcc.Input(id="tree_random_state", type="number", placeholder="input with range",min=1,max=42, step=1,value=42),html.Br(),html.Br(),
                              ],width=1
                           )
                        ]
                    ),
                   dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Centrer réduire",  html_for="tree_centrer_reduire",style={'font-weight': 'bold'}),
                                ], width=3
                            ),
                            dbc.Col(
                                [
                                dbc.Checklist(
                                    id="tree_centrer_reduire",
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
                                        id='tree_shuffle',
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
                                       html.B("stratify "),html.I("par defaut stratify=False"),html.Br(),html.P("Si ce n'est pas False, les données sont divisées de manière stratifiée en utilisant les étiquettes de la classe à prédire", className="card-text"),
                                    ], width=3
                                ),
                                dbc.Col(
                                   [
                                       dcc.Dropdown(
                                           id='tree_stratify',
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

            html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),

            html.Div([

                html.Div([

                    html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),
                    html.B("GridSearchCV_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                    dcc.Input(
                        id="Tree_GridSearchCV_number_of_folds",
                        type="number",
                        placeholder="input with range",
                        min=1,max=100, step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("GridSearchCV_scoring "),
                    html.I("par défaut = 'f1_macro'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour l'optimisation des hyperparamètres."),
                    dcc.Dropdown(
                        id='Tree_GridSearchCV_scoring',
                        options=[
                            {'label': "accuracy", 'value': "accuracy"},
                            {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                            {'label': "neg_brier_score", 'value': "neg_brier_score"},
                            {'label': "f1_micro", 'value': "f1_micro"},
                            {'label': "f1_macro", 'value': "f1_macro"},
                            {'label': "f1_weighted", 'value': "f1_weighted"},
                            #{'label': "neg_log_loss", 'value': "neg_log_loss"},
                            {'label': "precision_micro", 'value': "precision_micro"},
                            {'label': "precision_macro", 'value': "precision_macro"},
                            {'label': "precision_weighted", 'value': "precision_weighted"},
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
                        id="Tree_GridSearchCV_njobs",
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
                    dbc.Button("valider GridSearchCV", color="info",id='Tree_button_GridSearchCV',n_clicks=0),
                    dcc.Loading(id="ls-loading-0_tree", children=[html.Div(id="ls-loading-output-0_tree")], type="default"),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),
                    # on passe à une autre section (modification des paramètre) >> propre à la fonction
                    html.H4(html.B("Performance du modèle sur le jeu de test :")),
                    html.Br(),
                    html.B("criterion : ")," ",html.Br(),
                    dcc.Dropdown(
                        id='criterion',
                        options=[
                            {'label': 'gini', 'value': 'gini'},
                            {'label': 'entropy', 'value': 'entropy'}
                        ],
                        value='gini'
                    ),
                    html.Br(),
                    html.B("splitter : "),"",html.Br(),
                    dcc.Dropdown(
                        id='splitter',
                        options=[
                            {'label':'best','value':'best'},
                            {'label':'random','value':'random'}],
                        value="best",
                    ),
                    html.Br(),
                    html.B("max_depth : "),"",html.Br(),
                    dcc.Input(
                        id='max_depth',
                        type='number',
                        min=0,
                        max=10,
                        value=0,
                    ),
                    html.Br(),html.Br(),
                    html.B("min_samples_split : "),"",html.Br(),
                    dcc.Input(
                        id='min_samples_split',
                        type="number",
                        value=2,
                        max=10,
                        min=1
                    ),
                    html.Br(),html.Br(),
                    html.B("min_samples_leaf : "),"",html.Br(),
                    dcc.Input(
                        id='min_samples_leaf',
                        type='number',
                        min=1,
                        max=10,
                        value=1
                    ),
                    html.Br(),html.Br(),
                    html.B("max_leaf_nodes : "),"",html.Br(),
                    dcc.Input(
                        id='max_leaf_nodes',
                        type='number',
                        max=10,
                        min=0,
                        value=0
                    ),
                    html.Br(),html.Br(),
                    dbc.Button("Valider Fit & Predict", color="danger",id='Tree_button_FitPredict',n_clicks=0),
                    dcc.Loading(id="ls-loading-1_tree", children=[html.Div(id="ls-loading-output-1_tree")], type="default"),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),
                    # on passe à une autre section (validation croissé)
                    html.H4(html.B("Validation croisée :")),html.Br(),
                    html.B("cv_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                    dcc.Input(
                        id="Tree_cv_number_of_folds", type="number",
                        placeholder="input with range",min=1,max=100,
                        step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("cv_scoring "),html.I("par défaut = 'f1_macro'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                    dcc.Dropdown(
                        id='Tree_cv_scoring',
                        options=[
                            {'label': "accuracy", 'value': "accuracy"},
                            {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                            #{'label': "neg_brier_score", 'value': "neg_brier_score"},
                            {'label': "f1_micro", 'value': "f1_micro"},
                            {'label': "f1_macro", 'value': "f1_macro"},
                            {'label': "f1_weighted", 'value': "f1_weighted"},
                            {'label': "neg_log_loss", 'value': "neg_log_loss"},
                            {'label': "precision_micro", 'value': "precision_micro"},
                            {'label': "precision_macro", 'value': "precision_macro"},
                            {'label': "precision_weighted", 'value': "precision_weighted"},
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
                    dbc.Button("Valider Decision Tree CrossValidation", color="success",id='Tree_button_CrossValidation',n_clicks=0),
                    dcc.Loading(id="ls-loading-2_tree", children=[html.Div(id="ls-loading-output-2_tree")], type="default")

                ],className="six columns",style={"margin":10}),

                html.Div([

                    html.H4(html.B("Résultats :")),
                    html.Div(id="res_Tree_GridSearchCV"),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),html.Br(),
                    html.Div(id="res_Tree_FitPredict"),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),html.Br(),
                    html.Div(id="res_Tree_CrossValidation"), html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),
                    html.H4("Affichage de l'arbre de décision : "),html.Br(),
                    dbc.Button("Affichage de l'arbre",color="success",id='tree_plot_button',n_clicks=0)


                ],className="six columns",style={"margin":10}),

            ], className="row"),
        ],body=True
)
