#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:33:05 2024

@author: pedror
"""

import pandas as pd #toolbox to work with dataframes
import numpy as np #toolbox to work with narrays
import matplotlib.pyplot as plt #toolbox to do plots
from sklearn.svm import SVC #load the support vector machine model functions
from sklearn.model_selection import train_test_split #load the function to split train and test sets
from sklearn import metrics # get the report
from sklearn.metrics import classification_report # get the report
from sklearn import preprocessing # normalize the features
from sklearn.preprocessing import MinMaxScaler # normalize the features
from sklearn.feature_selection import SelectKBest #load the feature selector model  
from sklearn.feature_selection import chi2 #feature selector algorithm
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

def normalized_data (df,t):

    if (t==1):
        d=df.copy() # min max normalization
        for each_collum in range(0,df.shape[1]):
            max =df.iloc[:,each_collum].max()
            min =df.iloc[:,each_collum].min()
            d.iloc[:,each_collum]=(d.iloc[:,each_collum]-min)/(max-min)
    elif (t==2):
        d=df.copy() # mean normalization
        for each_collum in range(0,df.shape[1]):
            max =df.iloc[:,each_collum].max()
            min =df.iloc[:,each_collum].min()
            mean =df.iloc[:,each_collum].mean()
            d.iloc[:,each_collum]=(d.iloc[:,each_collum]-mean)/(max-min)
    
    else:
        d=df.copy() # standardization
        for each_collum in range(0,df.shape[1]):
            mean =df.iloc[:,each_collum].mean()
            std =df.iloc[:,each_collum].std()
            d.iloc[:,each_collum]=(d.iloc[:,each_collum]-mean)/(std)

    return d

    # Load the function to performe feature selection
from sklearn.feature_selection import SelectKBest
#load the feature selection algorithms
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import GenericUnivariateSelect

# function to do feature selection
def feature_selector(X_train,y_train,type,i):
    if (type == 1):
#ANOVA F-value between label/feature for classification tasks.
        bestfeatures = SelectKBest(score_func = f_classif, k=i)
    elif(type == 2):
#Mutual information for a discrete target.
        bestfeatures = SelectKBest(score_func=mutual_info_classif, k=i)
    elif(type == 3):
    #Chi-squared stats of non-negative features for classification tasks.
        bestfeatures = SelectKBest(score_func=chi2, k=i)
    elif(type == 4):
#Select features based on an estimated false discovery rate.
        bestfeatures = SelectKBest(score_func=SelectFdr, k=i)
    elif(type == 5):
#Select features based on family-wise error rate.
        bestfeatures = SelectKBest(score_func=SelectFwe, k=i)
#Perform the feature based on selected algorithm
    fit = bestfeatures.fit(X_train,y_train)
    cols_idxs = fit.get_support(indices=True)
    Xt=X_train.iloc[:,cols_idxs] # extract the best features for training
    return Xt,cols_idxs

# 3rd step - load and design the classifiers

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier,ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
classifiers = [
    SVC(gamma='auto',probability=True),
    KNeighborsClassifier(),
    LogisticRegression(solver='lbfgs'),
    BaggingClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    OneVsRestClassifier(LinearSVC(random_state=0)),
]
# Cross validation
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
from sklearn.metrics import roc_curve,auc
from scipy import interp
def cross_validation(model, _X, _y):
    scoring = {'accuracy' : make_scorer(accuracy_score), 
       'precision' : make_scorer(precision_score, average='macro'),
       'recall' : make_scorer(recall_score, average='macro'), 
       'f1_score' : make_scorer(f1_score, average='macro')}
    #_scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #kfold = model_selection.LeaveOneOut()  
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=stratified_cv,
                               scoring=scoring,
                               return_train_score=True)
    
    return [results['train_accuracy'].mean()*100, results['train_precision'].mean(), results['train_recall'].mean(), results['train_f1_score'].mean(),results['test_accuracy'].mean()*100,results['test_precision'].mean(), results['test_recall'].mean(),results['test_f1_score'].mean()]
t=['NIR','FTIR']

for u in t:

    df = pd.read_excel('Camellia_Data.xlsx',sheet_name=u) 
    target=df.iloc[:,-1]
    df=df.iloc[:,:-1]
    df.index=df['Unnamed: 0']
    df=df.drop('Unnamed: 0',axis=1)
    d_n=normalized_data (df.astype('Float64'),3)
    perf_results=pd.DataFrame()
    o=0
    for k in range(1,30,1):
        pcadata = PCA(n_components=k)
        for j in tqdm(range(30,d_n.shape[1],10)):    
            df_nf,cols_idxs=feature_selector(d_n,target,1,j)
            principalComponents = pcadata.fit_transform(df_nf)
            for i in classifiers:
                a,b,c,d,e,f,g,h=cross_validation(i,pd.DataFrame(principalComponents), target)
                perf_results[o]=[k,j,df.columns[cols_idxs],i,a,b,c,d,e,f,g,h]
                o=o+1       
        perf_results=perf_results.T
        perf_results.columns=['Pca_number_feat','# features','features','classifier',"Mean Training Accuracy","Mean Training Precision","Mean Training Recall","Mean Training F1 Score","Mean Validation Accuracy","Mean Validation Precision","Mean Validation Recall", "Mean Validation F1 Score"]
        perf_results.to_excel(f'/resultadosPCA{k}_{u}.xlsx')  
        perf_results=pd.DataFrame()