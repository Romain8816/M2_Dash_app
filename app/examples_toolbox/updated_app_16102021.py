# -*- coding: utf-8 -*-

#!! IMPORT SECTION --------------------------------------------------------------------------------------------------------------------
import base64
import datetime
import io
import dash
from dash import dash_table
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import plotly.express as px
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster


#!! CODE SECTION --------------------------------------------------------------------------------------------------------------------
app = dash.Dash(suppress_callback_exceptions=True)

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    return fig

# layout --------------------------------------------------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Upload(
            id='upload-data',
            children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
            ]),
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
            multiple=True),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.Div(id='output-data-upload'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='crossfilter-indicator-scatter',style={'width': '90vh', 'height': '90vh'},figure = blank_fig()
    ),
    html.Br(),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.Br(),
    html.Div(id='output-dropdown-X'),
    html.Br(),
    html.Div(id='output-X'),
    html.Br(),
    html.Div(id='output-X-info'),
    html.Br(),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.Br(),
    html.Div(id='output-dropdown-Y'),
    html.Br(),
    html.Div(id='output-Y'),
    html.Br(),
    html.Div(id='output-Y-info'),
    html.Br(),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.Br(),
    html.Div(id='marchine-learning-type'),
    html.Br(),
    html.Br(),
    html.Div(id='parameters'),
    html.Div(id=''),
    html.Br(),
    html.Div(id='kmeans-prediction'),
    html.Div(id='cah-prediction'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='pca-kmeans',style={'width': '90vh', 'height': '90vh'},figure = blank_fig()
    )
])

# methods --------------------------------------------------------------------------------------------------------------------
"""
# return input file as panda.dataframe
"""
def parse_content(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

"""
# convert panda.dataframe in datatable format
"""
def make_datatable(dataframe):
    return html.Div([

        dash_table.DataTable(
            id="tab",data = dataframe.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dataframe.columns],
            #style_table={'height': '300px', 'width':'600px', 'overflowY': 'auto'},
            sort_action='native',
            sort_mode='single',
            sort_by=[],
            page_action="native",
            page_current= 0,
            page_size= 10,
            style_cell_conditional=[
                {'if': {'column_id': c},'textAlign': 'center'} for c in dataframe.columns],
            )
    ])

# methods callback --------------------------------------------------------------------------------------------------------------------
@app.callback(Output('output-data-upload', 'children'), # update le tableau
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_table(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dataframe = [
            parse_content(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        datatable = make_datatable(dataframe[0])
        return datatable


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'), # update de la PCA
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
      State('upload-data', 'last_modified')])
def update_graph(input_X,input_Y,list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        dataframe = [
            parse_content(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        pca = PCA(n_components=2)
        temp = pca.fit_transform(dataframe[0][input_X])
        coord = pd.DataFrame(temp,columns=["comp1","comp2"])
        Y = pd.DataFrame(dataframe[0][input_Y],columns=["species"])
        result = pd.concat([coord,Y], axis=1)
        fig = px.scatter(result, x="comp1", y="comp2", color="species", hover_data=['species'],
                         title="PCA du jeu de données '{}'".format(list_of_names[0]))
        return fig



@app.callback(Output('output-dropdown-X', 'children'),  # update la liste déroulante des variables explicatives
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
                State('upload-data', 'last_modified')])
def update_dropdown_X(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dataframe = [
            parse_content(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return html.Div([
                 "Variables explicatives : ",
                 dcc.Dropdown(id='dropdown-X', multi=True, placeholder="selectionner les colonnes", value =dataframe[0].columns[0:len(dataframe[0].columns)-1],
                 options=[{'label': i, 'value': i} for i in dataframe[0].columns])
        ])

@app.callback(
    Output(component_id='output-X', component_property='children'), # update l'affichage des variables explicatives choisies et permet de les récupérer
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'))
def get_X(input_X,input_Y):
    if input_X != None and input_Y != None:
        if input_Y in input_X:
            return html.Div("/!\ X dans Y")
        else:
            return 'Output X: {} | Output Y: {}'.format(input_X,input_Y)
    else:
        return 'Output X: {} | Output Y: {}'.format(input_X,input_Y)

@app.callback(
    Output(component_id='output-X-info', component_property='children'), # update l'affichage des types des variables explicatives choisies
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
      State('upload-data', 'last_modified')])
def get_X_info(input_X,input_Y,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dataframe = [
            parse_content(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        datatable = make_datatable(dataframe[0][list(input_X)])
        if input_X != None and input_Y != None:
            if input_Y in input_X:
                return html.Div([ "" ])
            else:
                l = [datatable,html.Br(),html.Br(),"Type des variables :",html.Br()]
                dtype = "{}".format(dataframe[0][input_X].dtypes)
                dtype = dtype.replace("    ","  ---->  ")
                dtype = dtype.split("\n")
                for i in range(0,len(dtype)-1):
                    l.append("{}".format(dtype[i]))
                    l.append(html.Br())
                return(l)

@app.callback(Output('output-dropdown-Y', 'children'),  # update la liste déroulante des variables prédictives
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
                State('upload-data', 'last_modified')])
def update_dropdown_Y(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dataframe = [
            parse_content(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return html.Div([
                 "Variable à prédire : ",
                 dcc.Dropdown(id='dropdown-Y', multi=False, placeholder="selectionner une colonne", value =dataframe[0].columns[len(dataframe[0].columns)-1],
                 options=[{'label': i, 'value': i} for i in dataframe[0].columns])
        ])

@app.callback(
    Output(component_id='output-Y', component_property='children'), # update l'affichage de la variable prédictive choisie et permet de la récupérer
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'))
def get_Y(input_X,input_Y):
    if input_X != None and input_Y != None:
        if input_Y in input_X:
            return html.Div("/!\ Y dans X")
        else:
            return 'Output X: {} | Output Y: {}'.format(input_X,input_Y)
    else:
        return 'Output X: {} | Output Y: {}'.format(input_X,input_Y)


@app.callback(
    Output(component_id='output-Y-info', component_property='children'), # update l'affichage du type de la variable prédictive choisie
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
      State('upload-data', 'last_modified')])
def get_Y_info(input_X,input_Y,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dataframe = [
            parse_content(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        datatable = make_datatable(dataframe[0][input_Y].reset_index())
        if input_X != None and input_Y != None:
            if input_Y in input_X:
                return html.Div([ "" ])
            else:
                return html.P([datatable,html.Br(),html.Br(),"Type de la variable :",html.Br(),"{}  ---->  {}".format(input_Y,dataframe[0][input_Y].dtypes)])

@app.callback(
    Output('marchine-learning-type', 'children'), # update et récupère la méthode de ML choisie
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
      State('upload-data', 'last_modified')]
    )
def ml_methods(input_X,input_Y,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dataframe = [
            parse_content(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        Y_dtype = "{}".format(dataframe[0][input_Y].dtypes)
        if Y_dtype == "object":
            return html.Div([
                     "Algorithmes de machine learning :",
                     dcc.Dropdown(id='algo-type', multi=False, placeholder="selectionner un algorithme", persistence = True,
                     options=[{'label': i, 'value': i} for i in ["kmeans","cah"]])
            ])
        else:
            return ""

@app.callback(Output('parameters', 'children'), # update l'affiche des paramètres pour un model ML donnée
              Input('algo-type', 'value'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
                State('upload-data', 'last_modified')])
def ml_parameters(algo,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        if algo == "kmeans":
            return html.Div([ "Clusters",dcc.Slider(id='k-slider',min=1,max=10,step=1,value=3),html.Div(id='slider-output-container')])
        if algo == "cah":
            metric = html.Div([ "Metric : ", dcc.Dropdown(id='dropdown-metric', multi=False, placeholder="selectionner une metric", value ="euclidean",options=[{'label': i, 'value': i} for i in ["euclidean"]]) ])
            method = html.Div([ "Method : ", dcc.Dropdown(id='dropdown-method', multi=False, placeholder="selectionner une method", value ="ward",options=[{'label': i, 'value': i} for i in ["ward"]]) ])
            t = html.Div([ "Clusters : ", dcc.Slider(id='slider-t',min=1,max=10,step=1,value=3),html.Div(id='slider-t-output-container')])
            criterion = html.Div([ "Criterion : ", dcc.Dropdown(id='dropdown-criterion', multi=False, placeholder="criterion", value ="maxclust",options=[{'label': i, 'value': i} for i in ["maxclust"]]) ])

            return html.P([metric,html.Br(),method,html.Br(),t,html.Br(),criterion])

@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'), # update l'affichage du paramètres k (kmeans)
    [dash.dependencies.Input('k-slider', 'value'),
    dash.dependencies.Input('algo-type', 'value'),
    ])
def show_params_k(value,algo):
    #if algo == "kmeans":
    return 'KMeans nombre de clusters "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('slider-t-output-container', 'children'), # update l'affichage du paramètres t (cah)
    [dash.dependencies.Input('slider-t', 'value'),
    dash.dependencies.Input('algo-type', 'value'),
    ])
def show_params_y(value,algo):
    #if algo == "cah":
    return 'cah nombre de clusters "{}"'.format(value)

@app.callback(
    Output('kmeans-prediction', 'children'), # update l'affichage des prédictions (kmeans)
    Input('algo-type','value'),
    Input('k-slider', 'value'),
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
      State('upload-data', 'last_modified')]
    )
def predict_kmeans(algo,value,input_X,input_Y,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        if algo == "kmeans":
            dataframe = [
                parse_content(c, n, d)
                for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
            ]
            kmeans = KMeans(n_clusters=value)
            kmeans.fit(dataframe[0][input_X])
            y = list(dataframe[0][input_Y].replace({"setosa":1,"versicolor":2,"virginica":3}))
            rand_score = adjusted_rand_score(kmeans.labels_,y)
            return html.P(["y_pred : {}".format(kmeans.labels_),html.Br(),html.Br(),"y_target : {}".format(y),html.Br(),html.Br(),"rand score : {}".format(rand_score)])

@app.callback(
    Output('cah-prediction', 'children'), # update l'affichage des prédictions (cah)
    Input('algo-type','value'),
    Input('dropdown-metric', 'value'),
    Input('dropdown-method', 'value'),
    Input('slider-t', 'value'),
    Input('dropdown-criterion', 'value'),
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
      State('upload-data', 'last_modified')]
    )
def predict_cah(algo,metric,method,t,criterion,input_X,input_Y,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        if algo == "cah":
            dataframe = [
                parse_content(c, n, d)
                for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
            ]
            Z = linkage(dataframe[0][input_X],method=method,metric=metric)
            group_cah = fcluster(Z,t=t,criterion=criterion)
            y = list(dataframe[0][input_Y].replace({"setosa":1,"versicolor":2,"virginica":3}))
            rand_score = adjusted_rand_score(group_cah,y)
            return html.P(["y_pred : {}".format(group_cah),html.Br(),html.Br(),"y_target : {}".format(y),html.Br(),html.Br(),"rand score : {}".format(rand_score)])


"""
/!\ tentative de PCA (kmeans): fonctionnelle mais pas bien gérée en fonction de la méthode de ML choisie
"""
@app.callback(
    Output('pca-kmeans', 'figure'), # update de la figure kmeans
    State('algo-type','value'),
    Input('k-slider', 'value'),
    Input(component_id='dropdown-X', component_property='value'),
    Input(component_id='dropdown-Y', component_property='value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
      State('upload-data', 'last_modified')]
    )
def PCA_kmeans(algo,k,input_X,input_Y,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        if algo == "kmeans":
            dataframe = [
                parse_content(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]

            kmeans = KMeans(n_clusters=k)
            kmeans.fit(dataframe[0][input_X])

            pca = PCA(n_components=2)
            temp = pca.fit_transform(dataframe[0][input_X])
            coord = pd.DataFrame(temp,columns=["comp1","comp2"])
            Y = pd.DataFrame(list(map(str,kmeans.labels_)),columns=["kmeans_clusters"])
            result = pd.concat([coord,Y], axis=1)
            fig = px.scatter(result, x="comp1", y="comp2", color="kmeans_clusters", hover_data=['kmeans_clusters'],
                             title="PCA du jeu de données '{}' colorié par clusters du KMeans".format(list_of_names[0]))
            return fig
        else:
            return None





if __name__ == '__main__':
    app.run_server(debug=True)

#!! END --------------------------------------------------------------------------------------------------------------------
