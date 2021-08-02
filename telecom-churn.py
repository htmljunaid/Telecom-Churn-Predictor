# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:24:34 2021

@author: Junaid - M04000018
"""

#%% Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sklearn
from sklearn.tree import DecisionTreeClassifier
from dmba import plotDecisionTree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


#%% Setting up Data

data = pd.read_csv('cellphone.csv')
data.info()

#%% X and Y variables

y = data['Churn']

X = data.drop(['Churn','DataPlan','RoamMins','DayCalls'],axis=1)

X.info()

#%% Partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=18)

#%% Normalization

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
XNtrain = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
XNtest = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

#%% The best k?

resList=[]
for k in range(1,25):
    knn_bk = KNeighborsClassifier(n_neighbors=k)
    knn_bk.fit(XNtrain,y_train)
    y_pred = knn_bk.predict(XNtest)
    acc = metrics.accuracy_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred, pos_label=1)
    resList.append([k,acc,rec])
colsRes = ['k','Accuracy','Recall_Pos']
results = pd.DataFrame(resList, columns=colsRes)
print(results)
''' 3 neigbours are giving the most accurate result'''

#%% k=3 train and test

knn18 = KNeighborsClassifier(n_neighbors=3)
knn18.fit(XNtrain,y_train)

y_knn18 = knn18.predict(XNtest)


#%% Evaluating kNN

kappa = metrics.cohen_kappa_score(y_test,y_knn18)
print(kappa)
#kappa is 0.62

#Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_knn18)
cmPlot = metrics.ConfusionMatrixDisplay(cm)
cmPlot.plot(cmap='YlOrRd')
cmPlot.figure_.savefig('figs/CM_kNN.png')

#Classification Report
creport_KNN = metrics.classification_report(y_test, y_knn18)
print(creport_KNN)

#%%   Cart model

#%% The Best Depth?
#train
ctree_full = DecisionTreeClassifier(random_state=18)
ctree_full.fit(X_train, y_train)
plotDecisionTree(ctree_full, feature_names = X.columns)

# parameter setting
param_grid = {'max_depth':[1,2,3,4,5],
              'min_samples_split':[5,10,15,20],
              'min_impurity_decrease':[0, 0.0005, 0.001, 0.05, 0.01] }

gridSearch = GridSearchCV(ctree_full,
                          param_grid,
                          scoring = 'recall',
                          n_jobs=-1)

gridSearch.fit(X_train,y_train)
print("Best Recall:", gridSearch.best_score_)

print("Best parameters:", gridSearch.best_params_)

ctree_best = gridSearch.best_estimator_
plotDecisionTree(ctree_best, feature_names = X.columns)
#Best depth= 5

ctree18 = DecisionTreeClassifier(random_state=18, max_depth=5)
ctree18.fit(X_train, y_train)
plotDecisionTree(ctree18,
                 feature_names = X.columns,
                 class_names=ctree18.classes_)

#%% TEST CART Model

y_ctree18 = ctree18.predict(X_test)

#%% Evaluating CART Model

kappa = metrics.cohen_kappa_score(y_test,y_ctree18)
print(kappa)
#kappa is 0.57

#Confusion Matrix
cm_CART = metrics.confusion_matrix(y_test, y_ctree18)
cmPlot_CART = metrics.ConfusionMatrixDisplay(cm_CART)
CART_plot=cmPlot_CART.plot(cmap='YlOrRd')
CART_plot.figure_.savefig('FIGS/CM_CART.png')

#Classification Report
creport_CART = metrics.classification_report(y_test, y_ctree18)
print(creport_CART)


#%% Random Forest

rf_18 = RandomForestClassifier(n_estimators=500,random_state=18)
rf_18.fit(X_train,y_train)
importances = rf_18.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_18.estimators_], axis=0)
df = pd.DataFrame({'feature': X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance', ascending=False)
print(df)
ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')
plt.show()

y_rf_18 = rf_18.predict(X_test)

#%% Evaluating Random Forest

kappa = metrics.cohen_kappa_score(y_test,y_rf_18)
print(kappa)
# kappa is 0.66

#Confusion Matrix
cm_RF = metrics.confusion_matrix(y_test, y_rf_18)
cmPlot_RF = metrics.ConfusionMatrixDisplay(cm_RF)
a_RF=cmPlot_RF.plot(cmap='YlOrRd')
a_RF.figure_.savefig('figs/CM_RF.png')
#Classification Report
creport_RF = metrics.classification_report(y_test, y_rf_18)
print(creport_RF)

