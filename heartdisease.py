#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:15:01 2018

@author: manaswini
"""
import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.tree import tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import metrics
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn import cross_validation  #cross validation 
from urllib.request import urlopen # Get data from UCI Machine Learning Repository
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as pt

plt.style.use('ggplot')
pt.set_credentials_file(username='mannu', api_key='')

#function for normalization
def normalization(disease,tonormalize):
    dfnorm=disease.copy()
    for i in disease.columns:
        if(i in tonormalize):
            maxv=disease[i].max();
            minv=disease[i].min();
            dfnorm[i]=(disease[i]-np.mean(disease[i]))/(maxv-minv);
    return dfnorm






Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
Hungarian_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Switzerland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
#ClevelandHeartDisease = pd.read_csv(urlopen(Cleveland_data_URL), names = names) #gets Cleveland data
HungarianHeartDisease = pd.read_csv(urlopen(Hungarian_data_URL), names = names) #gets Hungary data
array = HungarianHeartDisease.values
l= array[:,0:13]
m=array[:,13]

#preprocessing

#SwitzerlandHeartDisease = pd.read_csv(urlopen(Switzerland_data_URL), names = names) #gets Switzerland data
datatemp = [HungarianHeartDisease] #combines all arrays into a list
#print(HungarianHeartDisease)

heartdisease = pd.concat(datatemp)#combines list into one array
print(heartdisease.head());
#print(heartDisease)
# replace ? with nan
heartdisease = heartdisease.replace('?',np.nan)
#check how many null values are present in the data
print(heartdisease.isnull().sum())
#delete the columns that contain many null values
del heartdisease['slope'];
del heartdisease['ca'];
del heartdisease['thal'];
#remove nan values for easy computation and store it in another variable
print("Checking null values\n");
heartDisease = heartdisease.dropna()
#check if any column still contain nan values
print(heartDisease.isnull().sum())
#chect the values by printing first 5 values
print(heartDisease.head())

#convert each item in heartdisease into numeric terms
for item in heartDisease: #converts everything to floats
    heartDisease[item] = pd.to_numeric(heartDisease[item])
#plot the box graph    
heartDisease.plot(kind='box',return_type='axes',color='green',sym='r*')
plt.show()
#plot the density graph
heartDisease.plot(kind="density");
plt.show();




#normalizing the data
#columns to normalize
n = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
dfnorm=normalization(heartDisease,n);
a=dfnorm.values;
X= a[:,0:10]
Y=a[:,10]
#check whether the function is applied by seeing first 5 values
print(dfnorm.head())
#plot the graphs and compare
dfnorm.plot(kind="density");
dfnorm.plot(kind="box",return_type="axes")
plt.show();

#check scatter matrix to see the relation between various column values
print("Checking relation by using Scatter matrix\n");

scatter_matrix(dfnorm,figsize=(10,10),diagonal='kde',color='black',s=100)

#TESTING ACCURACIES


#X=train['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak'];
#Y=test['num']

#testing accuracy without using any train test models.Predictions are done on the existing data itself

#K neighbours classification without using train_test split model
l=[];
knn=KNeighborsClassifier(n_neighbors=1);
knn.fit(X,Y)
y_pred=knn.predict(X);
l.append(["KNeigbors classifier n=1",metrics.accuracy_score(Y,y_pred)])
#K neighbours with 5 neighbours
knn=KNeighborsClassifier(n_neighbors=5);
knn.fit(X,Y)
y_pred=knn.predict(X);
l.append(["KNeigbors classifier n=5",metrics.accuracy_score(Y,y_pred)])


#logistic regression without using train_test split model
log=LogisticRegression();
log.fit(X,Y);
log.predict(X);
y_pred=log.predict(X)
l.append(["Logistic regression",metrics.accuracy_score(Y,y_pred)])

#using Decision tree regressor

model = DecisionTreeClassifier();
model.fit(X,Y)
y_pred=model.predict(X);
l.append(["Decision Tree Classifier",metrics.accuracy_score(Y,y_pred)])

#using RandomForestClassifier

model = RandomForestClassifier();
model.fit(X,Y)
y_pred=model.predict(X);
l.append(["Random Forest Classifier",metrics.accuracy_score(Y,y_pred)])

#creating a table
df = pd.DataFrame(l,columns=['Module','Accuracy'])



"""problem of not using train test method to split is that
1.They overfit
2.out of complex are not exactly estimated
"""

#now we use train_test_split function to split the data into train and test datasets
#check the shape of the data before splitting
print(X.shape)
print(Y.shape)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=4)
#check the shape of the data after splitting
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#applying models using training and testing data sets

#K neighbours with 1 neighbours
l1=[];
knn=KNeighborsClassifier(n_neighbors=1);
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test);
l1.append(["Kneighbors Classifier n=1",metrics.accuracy_score(y_test,y_pred)])

#K neighbours with 5 neighbours
knn=KNeighborsClassifier(n_neighbors=5);
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test);
l1.append(["Kneighbors Classifier n=5",metrics.accuracy_score(y_test,y_pred)])

#using Decision tree regressor
model = DecisionTreeClassifier();
model.fit(x_train,y_train)
y_pred=model.predict(x_test);
l1.append(["Decision Tree Classifier",metrics.accuracy_score(y_test,y_pred)])

#using random forest
print("random forest")
model = RandomForestClassifier();
model.fit(x_train,y_train)
y_pred=model.predict(x_test);
l1.append(["Random Forest Classifier",metrics.accuracy_score(y_test,y_pred)])

#let us get the better value for kneigbors algorithm

k=range(1,25);
scores=[]
for i in k:
    knn=KNeighborsClassifier(n_neighbors=i);
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test);
    scores.append(metrics.accuracy_score(y_test,y_pred));
plt.plot(k,scores)
plt.xlabel('k values');
plt.ylabel('accuracy');
plt.show();
#from graph consider the value for k as 18
#checking accuracy for k=18
knn=KNeighborsClassifier(n_neighbors=18);
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test);
l1.append(["KNeignors Classifier n=18",metrics.accuracy_score(y_test,y_pred)])

#logistic regression
log=LogisticRegression();
log.fit(x_train,y_train);
y_pred=log.predict(x_test)
l1.append(["Logisitc Regression",metrics.accuracy_score(y_test,y_pred)])

#Linear regression modl
d = pd.DataFrame(l1,columns=['Module','Accuracy'])

print("linear regression")
lin=LinearRegression();
model=lin.fit(x_train,y_train);
y_pred=model.predict(x_test);
#intecept in linear regression
print(lin.intercept_)
#coefficients for each of the feature values
for i,(a,b) in enumerate(zip(n,lin.coef_)):
    print(i,a,b);
#calculate root mean squared error in linear regression model
print("error in linear regression")
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)));
#print(metrics.accuracy_score(Y,y_pred))

#now cross validation 
#k fold cross validation is used to over come the disadvantages 
#of train/test split procedure
#for k neighbours classification model
l2=[]
knn=KNeighborsClassifier(n_neighbors=18);
l2.append(["KNeighbors Classifier n=18",cross_val_score(knn,X,Y,cv=10,scoring="accuracy").mean()])
#for logistic regression
log=LogisticRegression();
l2.append(["Logistic Regression",cross_val_score(log,X,Y,cv=10,scoring="accuracy").mean()]);
#for decision tree classifier
model = DecisionTreeClassifier();
l2.append(["DecisionTree Classifier",cross_val_score(model,X,Y,cv=10,scoring="accuracy").mean()]);
d1 = pd.DataFrame(l2,columns=['Module','Accuracy'])

#choose better tuning params using grid method and estimate the accuracy
k_range=np.arange(1,20)
knn=KNeighborsClassifier();
param_grid=dict(n_neighbors=k_range);
grid=GridSearchCV(knn,param_grid,cv=10,scoring="accuracy");
grid.fit(X,Y)
print(grid.best_score_);
print(grid.best_params_);
print(grid.best_estimator_);

#using confusion matrix to check the results 
#for knn model
print("CONFUSION MATRIX\n");
print("for knn model");
knn=KNeighborsClassifier(n_neighbors=18);
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test);
print(metrics.confusion_matrix(y_test,y_pred));
print("for logistic model")
log=LogisticRegression();
log.fit(x_train,y_train);
y_pred=log.predict(x_test)
print(metrics.confusion_matrix(y_test,y_pred));

#based upon the above observations we chose logistic regression model
#constructing ROC and AUC curves for logistic regression
log=LogisticRegression();
log.fit(x_train,y_train);
y_pred=log.predict(x_test)
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred);
plt.plot(fpr,tpr);
knn=KNeighborsClassifier(n_neighbors=18);
knn.fit(x_train,y_train);
y_prediction=knn.predict(x_test)
fp,tp,thresholds=metrics.roc_curve(y_test,y_prediction);
plt.plot(fp,tp);
model = DecisionTreeClassifier();
model.fit(x_train,y_train)
y_pre=model.predict(x_test);
f,t,thresholds=metrics.roc_curve(y_test,y_pre);
plt.plot(f,t);
plt.xlim([0.0,1.0]);
plt.ylim([0.0,1.0]);
plt.xlabel("False postive values");
plt.ylabel("True positive values");
plt.show();

print(df);
print("\n But better to use train test split method  The module and their accuracies are \n");
print(d);
print("\nUsing cross Validation the modules and their accuracies are \n");
print(d1);
print("\nBased upon the ROC curves and accuracy estimations Logistic regression is considered as the best model");











    


