import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")
from sklearn.pipeline import Pipeline

image_seg = pd.read_csv("Image_Segmention.csv")
X = image_seg.drop('Class', axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, le_y, stratify=le_y,
                                                    random_state=2022,
                                                    train_size=0.7)
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=3)
pipe = Pipeline([('STD',scaler),('KNN',knn)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#### Grid Search 
from sklearn.pipeline import Pipeline
scaler = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('STD',scaler),('KNN',knn)])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = { 'KNN__n_neighbors': np.arange(1,16)}
gcv = GridSearchCV(pipe, param_grid=params, 
                   scoring='neg_log_loss',cv=kfold)
gcv.fit(X, le_y)
print(gcv.best_params_)
print(gcv.best_score_)

##### Predicting on unlabelled data
tst_img = pd.read_csv("tst_img.csv")
best_model = gcv.best_estimator_
predictions = best_model.predict(tst_img)

print(le.inverse_transform(predictions))


######### Bankruptcy #################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'D', 'YR'],  axis=1)
y = brupt['D']

scaler = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('STD',scaler),('KNN',knn)])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = { 'KNN__n_neighbors': np.arange(1,16)}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3, 
                   scoring='roc_auc',cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### Predicting on unlabelled data
tst_brupt = pd.read_csv("testBankruptcy.csv")
tst_brupt.drop('NO', axis=1, inplace=True)
best_model = gcv.best_estimator_
predictions = best_model.predict(tst_brupt)
print(predictions)






