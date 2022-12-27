import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

hr = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))

######### Grid Search CV ##############
from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
lr = LogisticRegression(solver='saga',random_state=2022)
params = {'penalty':['l1','l2','elasticnet',None],
          'C': np.linspace(0.001, 4, 5),
          'l1_ratio': np.linspace(0.001, 1, 5)}

gcv = GridSearchCV(lr, param_grid=params, verbose=3,
                   scoring='roc_auc',cv=kfold)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

## with scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler = StandardScaler()
lr = LogisticRegression(solver='saga',random_state=2022)
pipe = Pipeline([('STD',scaler),('LR',lr)])
print(pipe.get_params())

params = {'LR__penalty':['l1','l2','elasticnet',None],
          'LR__C': np.linspace(0.001, 4, 5),
          'LR__l1_ratio': np.linspace(0.001, 1, 5)}

gcv = GridSearchCV(pipe, param_grid=params, verbose=3,
                   scoring='roc_auc',cv=kfold)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

############### Bankruptcy ######################
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv",index_col=0)
X = brupt.drop(['D','YR'], axis=1)
y = brupt['D']

scaler = StandardScaler()
lr = LogisticRegression(solver='saga',random_state=2022)
pipe = Pipeline([('STD',scaler),('LR',lr)])
print(pipe.get_params())

params = {'LR__penalty':['l1','l2','elasticnet',None],
          'LR__C': np.linspace(0.001, 4, 5),
          'LR__l1_ratio': np.linspace(0.001, 1, 5)}

gcv = GridSearchCV(pipe, param_grid=params, verbose=3,
                   scoring='roc_auc',cv=kfold)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

############## Image Segmentation ################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")
from sklearn.preprocessing import LabelEncoder
image_seg = pd.read_csv("Image_Segmention.csv")
X = image_seg.drop('Class', axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

scaler = StandardScaler()
lr = LogisticRegression(solver='saga',random_state=2022)
pipe = Pipeline([('STD',scaler),('LR',lr)])
print(pipe.get_params())

params = {'LR__penalty':['l1','l2','elasticnet',None],
          'LR__C': np.linspace(0.001, 4, 5),
          'LR__l1_ratio': np.linspace(0.001, 1, 5),
          'LR__multi_class':['ovr','multinomial']}

gcv = GridSearchCV(pipe, param_grid=params, verbose=3,
                   scoring='neg_log_loss',cv=kfold)
gcv.fit(X, le_y)

print(gcv.best_params_)
print(gcv.best_score_)

################### Vehicle ##########################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Vehicle Silhouettes")

vehicle = pd.read_csv("Vehicle.csv")
X = vehicle.drop('Class', axis=1)
y = vehicle['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

scaler = StandardScaler()
lr = LogisticRegression(solver='saga',random_state=2022)
pipe = Pipeline([('STD',scaler),('LR',lr)])
print(pipe.get_params())

params = {'LR__penalty':['l1','l2','elasticnet',None],
          'LR__C': np.linspace(0.001, 4, 5),
          'LR__l1_ratio': np.linspace(0.001, 1, 5),
          'LR__multi_class':['ovr','multinomial']}

gcv = GridSearchCV(pipe, param_grid=params, verbose=3,
                   scoring='neg_log_loss',cv=kfold)
gcv.fit(X, le_y)

print(gcv.best_params_)
print(gcv.best_score_)

