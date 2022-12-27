from sklearn.naive_bayes import GaussianNB
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
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = nb.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

############## K-Folds CV ###################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(nb, X, y, cv=kfold, scoring='roc_auc')
print(results.mean())

####################### Image Segmentation ####################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
image_seg = pd.read_csv("Image_Segmention.csv")
X = image_seg.drop('Class', axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, le_y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = nb.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

############## K-Folds CV ###################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(nb, X, le_y, 
                          cv=kfold, scoring='neg_log_loss')
print(results.mean())

