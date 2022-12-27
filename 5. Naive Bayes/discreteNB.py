import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Telecom")

telecom = pd.read_csv("Telecom.csv")
dum_tel = pd.get_dummies(telecom, drop_first=True)

X = dum_tel.drop('Response_Y', axis=1)
y = dum_tel['Response_Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

nb = BernoulliNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_pred_prob = nb.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))

############# cross_val_score ######################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
nb = BernoulliNB()
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
results = cross_val_score(nb, X, y, 
                          scoring='roc_auc', cv=kfold)
print(results.mean())

################ Cancer #############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer")

cancer = pd.read_csv("Cancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)

X = dum_cancer.drop('Class_recurrence-events', axis=1)
y = dum_cancer['Class_recurrence-events']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

nb = BernoulliNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_pred_prob = nb.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))

############# cross_val_score ######################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
nb = BernoulliNB()
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
results = cross_val_score(nb, X, y, 
                          scoring='roc_auc', cv=kfold)
print(results.mean())

