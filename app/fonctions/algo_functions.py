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
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score,f1_score,r2_score
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline

import time 

def train_test(X,y,test_size):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    return None

# Fonction qui permet d'instancier et fiter les modèles

# KMEANS
def build_kmeans(X,kmeans_n_clusters,kmeans_init,kmeans_n_init,
                kmeans_max_iter,kmeans_tol,kmeans_verbose,
                kmeans_random_state,kmeans_algorithm):
    kmeans = KMeans(n_clusters=kmeans_n_clusters,init=kmeans_init,n_init=kmeans_n_init,max_iter=kmeans_max_iter,
                    tol=kmeans_tol,verbose=kmeans_verbose,random_state=kmeans_random_state,algorithm=kmeans_algorithm)
    kmeans.fit(X)
    return kmeans

# SVM
def build_smv(X,param):
    clf = svm.SVC() # Linear Kernel
    clf.set_params()

def build_tree(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
    

    return tree
#marche pas pas compris le pipeline
def buid_linearReg_pipeline(fit_intercept,copy_X,n_jobs,centre_reduit):
    if centre_reduit == ["yes"]:
        numerical_pipeline = make_pipeline(SimpleImputer(),StandardScaler())
    else : 
        numerical_pipeline = make_pipeline(SimpleImputer())
    categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(drop='first',sparse=False))
    

    preprocessor = make_column_transformer((numerical_pipeline),
                                            (categorical_pipeline))

    model = make_pipeline(preprocessor,LinearRegression(fit_intercept=fit_intercept,copy_X=copy_X,n_jobs=n_jobs))

    return model

def buid_linearReg(fit_intercept,copy_X,n_jobs):
    linearReg =  LinearRegression(fit_intercept=fit_intercept,copy_X=copy_X,n_jobs=n_jobs)
    return linearReg

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
    elif clf == "Arbre de decision" : 
        t1 = time.time()
        grid_search = GridSearchCV(DecisionTreeClassifier(), params, scoring=scoring, cv=cv, n_jobs=njobs)
        grid_search = grid_search.fit(X,Y)
        t2 = time.time()
        diff = t2 - t1
        return [grid_search,diff]
    elif clf == "Regression lineaire" : 
        t1 = time.time()
        grid_search = GridSearchCV(LinearRegression(), params, scoring=scoring, cv=cv, n_jobs=njobs)
        grid_search = grid_search.fit(X,Y)
        t2 = time.time()
        diff = t2 - t1
        return [grid_search,diff]
    
def cross_val(clf,X,Y,cv,scoring):
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

def cross_validation(clf,X,Y,cv,scoring):
    if str(clf).startswith("DecisionTreeClassifier"):
        if scoring not in ["f1_binary","recall_binary","precision_binary","recall_micro","recall_macro","recall_weighted","precision_micro","precision_macro","precision_weighted"]:
            cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=scoring)
        else:
            if scoring == "f1_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(f1_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_micro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="micro"))
            if scoring == "recall_macro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="macro"))
            if scoring == "recall_weighted":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="weighted"))
            if scoring == "precision_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "precision_micro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="micro"))
            if scoring == "precision_macro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="macro"))
            if scoring == "precision_weighted":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="weighted"))
        return cross_val

    if str(clf).startswith("LinearRegression"):
        if scoring == "r2":
            cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(r2_score,greater_is_better=True))
        if scoring == "MSE":
            cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring="neg_mean_squared_error")
        return cross_val