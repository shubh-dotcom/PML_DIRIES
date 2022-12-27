from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)
### Linear
da = LinearDiscriminantAnalysis()
da.fit(X_train, y_train)

y_pred = da.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = da.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

############## K-Folds CV ###################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(da, X, y, cv=kfold, scoring='roc_auc')
print(results.mean())


### Quadratic
da = QuadraticDiscriminantAnalysis()
da.fit(X_train, y_train)

y_pred = da.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = da.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

############## K-Folds CV ###################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(da, X, y, cv=kfold, scoring='roc_auc')
print(results.mean())

##################### Vehicle ###############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Vehicle Silhouettes")
from sklearn.preprocessing import LabelEncoder
vehicle = pd.read_csv("Vehicle.csv")
X = vehicle.drop('Class', axis=1)
y = vehicle['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)

da = LinearDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(da, X, le_y, cv=kfold, scoring='neg_log_loss')
print(results.mean())

### ROC AUC OVR
da = LinearDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(da, X, le_y, cv=kfold, scoring='roc_auc_ovr')
print(results.mean())

da = QuadraticDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(da, X, le_y, cv=kfold, scoring='neg_log_loss')
print(results.mean())

################### Satellite ##########################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Satellite Imaging")
satellite = pd.read_csv("Satellite.csv", sep=";")
X = satellite.drop('classes', axis=1)
y = satellite['classes']

le = LabelEncoder()
le_y = le.fit_transform(y)

da = LinearDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(da, X, le_y, cv=kfold, scoring='neg_log_loss')
print(results.mean())


da = QuadraticDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(da, X, le_y, cv=kfold, scoring='neg_log_loss')
print(results.mean())

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(nb, X, le_y, cv=kfold, scoring='neg_log_loss')
print(results.mean())
