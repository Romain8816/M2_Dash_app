import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output
import os

app = dash.Dash(__name__)
app.title="Machine Learning App"

data_path = os.getcwd() +'\\data\\'
files = [f for f in os.listdir(data_path)]

app.layout = (html.Div(children=[
        html.H1('Machine Learning App',className='h1'),

        html.P('Application réalisée par Olivier IMBAUD, Inès KARA, Romain DUDOIT'),
    
        dcc.Dropdown(
            id='test',
            options=[
                {'label':i, 'value':i} for i in files
            ])
    ])
)

if __name__=='__main__':
    app.run_server(debug=True)