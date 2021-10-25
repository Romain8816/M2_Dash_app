import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Row import Row
from pkg_resources import NullProvider
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

def train_test(X,y,test_size):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    return None



# Fonction qui permet d'instancier et fiter les modèles
def build_kmeans(X,n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire):
    kmeans = KMeans(n_clusters=n_clusters,init=init,n_init=n_init,max_iter=max_iter,tol=tol,verbose=verbose,random_state=random_state,algorithm=algorithm)
    if centrer_reduire != None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    kmeans.fit(X)
    return kmeans


def build_smv(X,param):
    clf = svm.SVC() # Linear Kernel
    clf.set_params()