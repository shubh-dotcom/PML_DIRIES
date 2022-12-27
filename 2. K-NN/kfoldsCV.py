import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

mowers = pd.read_csv("RidingMowers.csv")

dum_mow = pd.get_dummies(mowers, drop_first=True)

X = dum_mow.drop('Response_Not Bought', axis=1)
y = dum_mow['Response_Not Bought']

knn = KNeighborsClassifier(n_neighbors=3)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
# Accuracy 
cross_val_score(knn, X, y, cv = kfold)
# ROC AUC
results = cross_val_score(knn, X, y, cv = kfold, scoring='roc_auc')
print(results.mean())

acc = []
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    results = cross_val_score(knn, X, y, cv = kfold, scoring='roc_auc')
    acc.append(results.mean())

i_max = np.argmax(acc)
best_k = Ks[i_max]
print("Best n_neigbors =", best_k)
print("Best Score =", acc[i_max])

#### Grid Search CV
from sklearn.model_selection import GridSearchCV
params = { 'n_neighbors': [1,3,5,7,9,11,13,15] }
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params, scoring='roc_auc',cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############# Boston ###################
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = { 'n_neighbors': np.arange(1,16)}
knn = KNeighborsRegressor()
gcv = GridSearchCV(knn, param_grid=params, scoring='r2',cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### Using Pipe with scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('STD',scaler),('KNN',knn)])
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = { 'KNN__n_neighbors': np.arange(1,16)}
knn = KNeighborsRegressor()
gcv = GridSearchCV(pipe, param_grid=params, scoring='r2',cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################# Medical Cost Expenses #######################
import os
from sklearn.linear_model import LinearRegression
os.chdir(r"Z:\PML\Cases\Medical Cost Personal")
insurance = pd.read_csv("insurance.csv")
dum_ins = pd.get_dummies(insurance, drop_first=True)

X = dum_ins.drop('charges', axis=1)
y = dum_ins['charges']

### Linear Regression
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = LinearRegression()
results = cross_val_score(lr, X, y, cv=kfold, scoring='r2')
print(results.mean())

### K-NN
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('STD',scaler),('KNN',knn)])
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = { 'KNN__n_neighbors': np.arange(1,16)}
knn = KNeighborsRegressor()
gcv = GridSearchCV(pipe, param_grid=params, scoring='r2',cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### Predicting on unlabelled data
knn = KNeighborsRegressor(n_neighbors=7)
pipe = Pipeline([('STD',scaler),('KNN',knn)])
pipe.fit(X,y)

tst_insure = pd.read_csv("tst_insure.csv")
dum_tst = pd.get_dummies(tst_insure, drop_first=True)
print(X.dtypes)
print(dum_tst.dtypes)
predictions = pipe.predict(dum_tst)

# or using Grid Search 
pd_cv = pd.DataFrame(gcv.cv_results_)
best_model = gcv.best_estimator_
tst_insure = pd.read_csv("tst_insure.csv")
dum_tst = pd.get_dummies(tst_insure, drop_first=True)
predictions = best_model.predict(dum_tst)



