import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydot
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz 
from IPython.display import Image
import pydotplus
from sklearn.model_selection import cross_val_score

#Retrieving the Dataset
data = pd.read_csv('winequality-white.csv', delimiter = ';')
print data.head(5)
print data.shape
print data.describe()
x=data.drop('quality', axis=1)
x_list = list(x.columns)
quality = data["quality"].values
Y=[]
target=[]

#Classifying the wines
for i in quality:
	if i<5:
		Y.append(0)#bad
		target.append("Bad")
	elif i>6:
		Y.append(1)#good
		target.append("Good")
	else:
		Y.append(2)#normal
		target.append("Normal")
y=np.asarray(Y)
count=[(i, target.count(i)) for i in set(target)]
print count
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20) 
scalar = preprocessing.StandardScaler()
X_train = scalar.fit_transform(x_train)
X_test = scalar.fit_transform(x_test)

#Random forest classifier
rf = RandomForestClassifier(n_estimators = 100,min_sample_split=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Accuracy=accuracy_score(y_pred,y_test) #Accuracy of the test dataset is calculated

#Without optimisation
print "RANDOM FOREST without optimisation"
print "Best parameters using grid search cross validation",rf.get_params()
print "\nAccuracy: "
print Accuracy
print "\nconfusion matrix"
y_test_re = list(y_test)
for i in range(len(y_test_re)):
    if y_test_re[i] == 0:
        y_test_re[i] = "bad"
    if y_test_re[i] == 1:
        y_test_re[i] = "good"
    if y_test_re[i] == 2:
        y_test_re[i] = "normal"
pred_dt1_re = list(y_pred)
for i in range(len(pred_dt1_re)):
    if pred_dt1_re[i] == 0:
        pred_dt1_re[i] = "bad"
    if pred_dt1_re[i] == 1:
        pred_dt1_re[i] = "good"
    if pred_dt1_re[i] == 2:
        pred_dt1_re[i] = "normal"
y_actu = pd.Series(y_test_re, name='Actual')
y_pred1 = pd.Series(pred_dt1_re, name='Predicted')
dt1_confusion = pd.crosstab(y_actu, y_pred1)
print(dt1_confusion)
print(classification_report(y_test, y_pred))

#With optimisation
param_dist={'max_depth':[2,3,4],'bootstrap':[True,False],'max_features':['auto','sqrt','log2',None],'criterion':['gini','entropy']}
cv_rf=GridSearchCV(rf,cv=10,param_grid=param_dist,n_jobs=3)
cv_rf.fit(X_train, y_train)
y_predcv = cv_rf.predict(X_test)
Accuracy=accuracy_score(y_predcv,y_test) #Accuracy of the test dataset is calculated
print "RANDOM FOREST with optimisation"
print "Best parameters using grid search cross validation",cv_rf.best_params_
print "\nAccuracy: "
print Accuracy
print "\nconfusion matrix"
y_test_re1 = list(y_test)
for i in range(len(y_test_re)):
    if y_test_re1[i] == 0:
        y_test_re1[i] = "bad"
    if y_test_re1[i] == 1:
        y_test_re1[i] = "good"
    if y_test_re1[i] == 2:
        y_test_re1[i] = "normal"
pred_dt1_re1 = list(y_predcv)
for i in range(len(pred_dt1_re)):
    if pred_dt1_re1[i] == 0:
        pred_dt1_re1[i] = "bad"
    if pred_dt1_re1[i] == 1:
        pred_dt1_re1[i] = "good"
    if pred_dt1_re1[i] == 2:
        pred_dt1_re1[i] = "normal"
y_actu1 = pd.Series(y_test_re1, name='Actual')
y_pred11 = pd.Series(pred_dt1_re1, name='Predicted')
dt1_confusion = pd.crosstab(y_actu1, y_pred11)
print(dt1_confusion)
print (confusion_matrix(y_predcv,y_test))
print(classification_report(y_test, y_predcv))

#Important features
imp= rf.feature_importances_
coef=np.sort(imp)[-5:]
print coef
co=np.argsort(imp)[-5:]
print co
outp=[]
for i in co:
	outp.append(x_list[i])
print outp
index=np.arange(len(outp))
li=imp.tolist()
print type(li),type(index)
plot.figure(1,figsize=(10,8))
plot.title("Normalized weights for First five most Predictive features")
plot.bar(co,coef)
plot.xticks(co, outp, fontsize=5, rotation=30)
plot.xlabel('Attributes')
plot.ylabel('Weights')

#with five predictive features
X_train_max=np.array(X_train[:,co],ndmin=2).T #The training set taking features with high weight coefficient
X_test_max=np.array(X_test[:,co],ndmin=2).T 
print X_train_max.shape
print X_test_max.shape
x_train_max=X_train_max.reshape(3918,5)
x_test_max=X_test_max.reshape(980,5)
rf1 = RandomForestClassifier(n_estimators = 100)
rf1.fit(x_train_max, y_train)
y_pred1 = rf1.predict(x_test_max)
Accuracy1=accuracy_score(y_pred1,y_test) #Accuracy of the test dataset is calculated
print "RANDOM FOREST with five most predictive feature"
print "Best parameters using grid serach cross validation",rf1.get_params()
print "\nAccuracy: "
print Accuracy1
print "\nconfusion matrix"
y_test_re2= list(y_test)
for i in range(len(y_test_re)):
    if y_test_re2[i] == 0:
        y_test_re2[i] = "bad"
    if y_test_re2[i] == 1:
        y_test_re2[i] = "good"
    if y_test_re2[i] == 2:
        y_test_re2[i] = "normal"
pred_dt1_re2 = list(y_pred1)
for i in range(len(pred_dt1_re)):
    if pred_dt1_re2[i] == 0:
        pred_dt1_re2[i] = "bad"
    if pred_dt1_re2[i] == 1:
        pred_dt1_re2[i] = "good"
    if pred_dt1_re2[i] == 2:
        pred_dt1_re2[i] = "normal"
y_actu2 = pd.Series(y_test_re2, name='Actual')
y_pred12 = pd.Series(pred_dt1_re2, name='Predicted')
dt1_confusion = pd.crosstab(y_actu2, y_pred12)
print(dt1_confusion)
print (confusion_matrix(y_pred1,y_test))
print(classification_report(y_test, y_pred1))

#Frequency of different qualities
plot.figure(2,figsize=(10,8))
plot.title("Frequency of different Qualities")
sns.countplot(data["quality"], palette="muted")
plot.xlabel('Class')
plot.ylabel('Frequency')
#print "coef",coef
plot.figure(3,figsize=(10,8))
plot.title("Frequency of different Quality classes")
sns.countplot(target, palette="muted")
plot.xlabel('Class')
plot.ylabel('Frequency')
print "type",type(x_list),type(target)
estimator = rf.estimators_[5]
dot_data=StringIO()

#Accuracy of different number of trees
scores=[]
i=[]
for val in range(30,500,10):
	cl=RandomForestClassifier(n_estimators=val)
	cl.fit(X_train, y_train)
	y_pred = cl.predict(X_test)
	Accurac=accuracy_score(y_pred,y_test)
	scores.append(Accurac)
	i.append(val)
plot.figure(4,figsize=(10,8))
plot.title("Accuracy for different number of trees")
plot.plot(i,scores)
print scores
print i
plot.xlabel('Number of trees')
plot.ylabel('Accuracy score')

plot.show()

