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
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score
import time

def train_test(X,y,test_size):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    return None

def get_best_params(X,Y,clf,params,cv,scoring,njobs):
    if clf == "KNeighborsClassifier":
        t1 = time.time()
        grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=scoring, cv=cv, n_jobs=njobs)
        if scoring == "recall_micro":
            grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(recall_score,average="micro"), cv=cv, n_jobs=njobs)
        if scoring == "recall_macro":
            grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(recall_score,average="macro"), cv=cv, n_jobs=njobs)
        if scoring == "recall_weighted":
            grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(recall_score,average="weighted"), cv=cv, n_jobs=njobs)
        if scoring == "precision_micro":
            grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(precision_score,average="micro"), cv=cv, n_jobs=njobs)
        if scoring == "precision_macro":
            grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(precision_score,average="macro"), cv=cv, n_jobs=njobs)
        if scoring == "precision_weighted":
            grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(precision_score,average="weighted"), cv=cv, n_jobs=njobs)
        grid_search = grid_search.fit(X.values,Y.values)
        t2 = time.time()
        diff = t2 - t1
        return [grid_search,diff]

def binariser(df,features,target):
    df_ = pd.get_dummies(df[features])
    f = df_.columns
    df = pd.concat([df_,df[target]], axis=1)
    return [df,f]

def centrer_reduire_norm(df,features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    X = pd.DataFrame(X,columns=features)
    return X

def build_smv(X,param):
    clf = svm.SVC() # Linear Kernel
    clf.set_params()

def build_KNeighborsClassifier(n_neighbors,weights,algorithm,leaf_size,p,metric):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
    return clf

def cross_validation(clf,X,Y,cv,scoring):
    t1 = time.time()
    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=scoring)
    if scoring == "recall_micro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="micro"))
    if scoring == "recall_macro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="macro"))
    if scoring == "recall_weighted":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="weighted"))
    if scoring == "precision_micro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="micro"))
    if scoring == "precision_macro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="macro"))
    if scoring == "precision_weighted":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="weighted"))
    t2 = time.time()
    diff = t2 - t1
    return [cross_val,diff]
