import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output
import os
import pandas as pd

app = dash.Dash(__name__)
app.title="Machine Learning App"

data_path = os.getcwd() +'\\data\\'
files = [f for f in os.listdir(data_path)]

app.layout = (html.Div(children=[
        html.H1('Machine Learning App',className='h1'),

        html.P('Application réalisée par Olivier IMBAUD, Inès KARA, Romain DUDOIT'),
        html.P('Sélectionner un jeu de données'),
        dcc.Dropdown(
            id='file_selection',
            options=[
                {'label':i, 'value':i} for i in files
            ]), 
        dcc.Dropdown(
            id='variable_cible'
        )
    ])
)

@app.callback(
    Output(component_id='variable_cible', component_property='options'),
    Input(component_id='file_selection', component_property='value')
)
def selection_variable_cible(file):
    df = pd.read_csv(data_path+'iris_data.csv')
    variable_list= list(df.columns.values)
    return ({'label':v, 'value':v} for v in variable_list)



if __name__=='__main__':
    app.run_server(debug=True)