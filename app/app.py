import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input,Output,State
import os
from dash.exceptions import PreventUpdate
from dash import dash_table
import numpy as np
from statistics import *
from fonctions.various_functions import *
from fonctions.algo_functions import *
from layout.layout import location_folder, dataset_selection, target_selection, features_selection, regression_tabs, classification_tabs
from fonctions.various_functions import allowed_files, get_pandas_dataframe
from fonctions.algo_functions import *

from callbacks import svr_callbacks, knnreg_callbacks, knnclas_callbacks, log_callbacks, tree_callbacks, lr_callbacks

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
app.title="Machine Learning App"


# VARIABLES
form = dbc.Form([location_folder, dataset_selection,target_selection,features_selection])

allowed_extensions =('.csv','.xlsx','.xls')



#************************************************************************ MAIN LAYOUT **********************************************************************
#***********************************************************************************************************************************************************
app.layout = html.Div(children=[
        html.Div(
            [
                html.H1('Réalisation d’une interface d’analyse de données par apprentissage supervisé'),
                html.H5(['Olivier IMBAUD, Inès KARA, Romain DUDOIT'],style={'color':'white','font-weight':'bold'}),
                html.H6('Master SISE (2021-2022)',style={'color':'white','font-weight':'bold'})
            ],className='container-fluid top'
        ),
        html.Br(),
        html.Div(
            [
                dbc.Row(
                    [
                        form,
                        html.Br(),
                        dbc.Col(html.Div(id='dataset'),width="100%"),
                        html.P(id='nrows',children="",className="mb-3"),
                    ]
                )
            ], className='container-fluid'
        ),
        html.Div(id='stats'),
        html.Div(
            [
                # Affichage des tabs, caché par défaut
                dbc.Collapse(
                    id='collapse_tab',
                    is_open=False
                ),

                dbc.RadioItems(
                    id="model_selection",
                ),
            ],
            className='container-fluid'
        ),
        html.Br(),
        html.Br(),
        dcc.Store(id='num_variables')
])

#*********************************************************************** CALLBACKS *************************************************************************
#***********************************************************************************************************************************************************

########################################################################################################
# (INIT) RECUPERATION DE LA LISTE DES FICHIERS AUTORISES DANS LE REPERTOIRE RENSEIGNE

@app.callback(
    Output('file_selection','options'), # mise à jour de la liste des fichiers dans le répertoire
    Input('validation_folder', 'n_clicks'), # valeur du bouton
    State(component_id="location_folder",component_property='value') #valeur de l'input
)
def update_files_list(n_clicks,data_path):
    # Si on a appuyer sur le bouton valider alors
    if n_clicks !=0:
        # On essaie de parcourir les fichiers dans le répertoire data_path
        try :
            files = os.listdir(r'%s' %data_path)
            filtred_files=allowed_files(data_path,allowed_extensions)
        # Si le répertoire n'existe
        except:
            return dash.no_update, '{} is prime!'.format(data_path)    #/!\ Exception à reprendre

        # --- /!\ return ([{'label':f, 'value':(r'%s' %(data_path+'\\'+f))} for f in filtred_files]) # WINDOWS
        return ([{'label':f, 'value':(r'%s' %(data_path+'/'+f))} for f in filtred_files]) # LINUX / MAC-OS
    else:
        raise PreventUpdate

########################################################################################################
# (INIT) LECTURE DU FICHIER CHOISIT ET MAJ DE LA DROPDOWN DES VARIABLES CIBLES

@app.callback(
    Output(component_id='target_selection', component_property='value'),
    Output(component_id='target_selection', component_property='options'),
    Output(component_id='dataset', component_property='children'),
    Output(component_id='num_variables', component_property='data'),
    Input(component_id='file_selection', component_property='value'),
)
def FileSelection(file_path):
    if file_path is None:
        raise PreventUpdate
    else:
        df = get_pandas_dataframe(file_path)
        variables = df.columns.tolist()
        num_variables = df.select_dtypes(include=np.number).columns.tolist()
        table =dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name":i,"id":i} for i in df.columns],
                fixed_rows={'headers': True},
                page_size=20,
                sort_action='native',
                sort_mode='single',
                sort_by=[],
                style_cell={'textAlign': 'left','minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_table={'height': '400px', 'overflowY': 'scroll','overflowX': 'scroll'},
                style_header={'backgroundColor': 'dark','fontWeight': 'bold'},
                style_cell_conditional=[
                    {'if': {'column_id': c},'textAlign': 'center'} for c in df.columns],
            )
        return (None,[{'label':v, 'value':v} for v in variables],table,num_variables)

########################################################################################################
# (INIT) CHARGEMENT DES VARIABLES EXPLICATIVES A SELECTIONNER

@app.callback(
        Output(component_id='features_selection', component_property='options'),
        Output(component_id='features_selection', component_property='value'),
        #Output(component_id='collapse_tab', component_property='is_open'),
        Input(component_id='target_selection', component_property='value'),   # valeur de la variable cible
        Input(component_id='target_selection', component_property='options'), # liste des variables cibles
        Input(component_id='features_selection', component_property='value')  # valeur des variables explicatives.
)
def TargetSelection(target,options,feature_selection_value):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné
    if target is None:
        return ([{'label':"", 'value':""}],None)
    else :
        variables = [d['value'] for d in options]
        if feature_selection_value == None:
            return (
                [{'label':v, 'value':v} for v in variables if v!=target],
                [v for v in variables if v!=target]
            )
        else:
            if len(feature_selection_value) >= 1:
                return (
                    [{'label':v, 'value':v} for v in variables if v!=target],
                    [v for v in feature_selection_value if v!=target]
                )
            else:
                return (
                    [{'label':v, 'value':v} for v in variables if v!=target],
                    [v for v in variables if v!=target]
                )


########################################################################################################
# (INIT) Proposition du/des modèles qu'il est possible de sélectionner selon le type de la variable cible

@app.callback(
    Output(component_id='collapse_tab',component_property='children'),    # Tab de classification ou de régression
    Output(component_id='collapse_tab',component_property='is_open'),     # Affichage des onglets
    Input(component_id='file_selection', component_property='value'),     # Emplacement du fichier
    Input(component_id='num_variables',component_property='data'),        # Liste des variables numérique
    Input(component_id='target_selection',component_property='value'),    # Variable cible
    Input(component_id='features_selection',component_property='value'),  # Variables explicatives
    Input(component_id='model_selection',component_property='value')      # Model choisit.
)
def ModelSelection(file,num_variables,target_selection,feature_selection,selected_model):
    # Si la variable cible à été sélectionné
    if target_selection != None:
        # Si la variable est numérique
        if target_selection in num_variables:
            return (
                regression_tabs,
                True
            )

        # Sinon (si la variable est qualitative)
        else:
            return (
                classification_tabs,
                True
            )
    # Sinon ne rien faire
    else:
        return ("",False)
        #raise PreventUpdate


########################################################################################################
# (Stats descriptives)

@app.callback(
    Output('stats','children'),
    Input('file_selection','value'),
    Input('features_selection','value'),
    Input('target_selection','value'),
    Input('num_variables','data'),
)
def stats_descrip(file,features,target,num_var):
    if None in (file,features,target):
        PreventUpdate
    else:
        df = get_pandas_dataframe(file)
        #X= df[features]
        #y= df[target]
        if target not in num_var :
            return dcc.Graph(
                figure = {
                    'data':[
                        {'x':df[target].value_counts().index.tolist(), 'y':df[target].value_counts().values.tolist(),'type': 'bar'}
                    ],
                    'layout': {
                        'title': 'Distribution de la variable '+target
                    }
                }
            )
        else :
            fig = px.histogram(df,x=target)
            return dcc.Graph(figure=fig)



########################################################################################################
# (Régression) SVM
svr_callbacks.Gridsearch(app)
svr_callbacks.FitPredict(app)
svr_callbacks.CrossValidation(app)


########################################################################################################
# (Classification) Régression logistique
log_callbacks.Gridsearch(app)
log_callbacks.FitPredict(app)
log_callbacks.CrossValidation(app)


########################################################################################################
# (Régression) KNN
knnreg_callbacks.Gridsearch(app)
knnreg_callbacks.FitPredict(app)
knnreg_callbacks.CrossValidation(app)

########################################################################################################
# (Classification) KNN
knnclas_callbacks.Gridsearch(app)
knnclas_callbacks.FitPredict(app)
knnclas_callbacks.CrossValidation(app)

###############################################################################
############### Decision Tree
tree_callbacks.Gridsearch(app)
tree_callbacks.FitPredict(app)
tree_callbacks.CrossValidation(app)

###############################################################################
############### Régression linéaire
lr_callbacks.Gridsearch(app)
lr_callbacks.FitPredict(app)
lr_callbacks.CrossValidation(app)


app.css.append_css({'external_url': './assets/style4columns.css' # LINUX - MAC-OS
})

if __name__=='__main__':
    app.run_server(debug=True)
