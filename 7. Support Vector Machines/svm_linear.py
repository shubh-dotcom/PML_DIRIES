import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import os
from sklearn.svm import SVC
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

svm = SVC(kernel='linear', probability=True, random_state=2022)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

#################### Grid Search CV #########################
from sklearn.model_selection import GridSearchCV

params = {'C': np.linspace(0.001, 10, 20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(kernel='linear', probability=True, random_state=2022)
gcv = GridSearchCV(svm, param_grid=params, cv=kfold, verbose=3,
                   scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

##### Using Scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
svm = SVC(kernel='linear', probability=True, random_state=2022)

pipe = Pipeline([('STD', scaler),('SVM',svm)])
print(pipe.get_params())

params = {'SVM__C': np.linspace(0.001, 10, 20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(kernel='linear', probability=True, random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3,
                   scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

################### Kyphosis ############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis")
kyph = pd.read_csv("Kyphosis.csv")
dum_ky = pd.get_dummies(kyph, drop_first=True)
X = dum_ky.drop('Kyphosis_present', axis=1)
y = dum_ky['Kyphosis_present']


scaler = StandardScaler()
svm = SVC(kernel='linear', probability=True, random_state=2022)

pipe = Pipeline([('STD', scaler),('SVM',svm)])
print(pipe.get_params())

params = {'SVM__C': np.linspace(0.001, 10, 20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(kernel='linear', probability=True, random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3,
                   scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
