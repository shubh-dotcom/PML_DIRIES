import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score 

boston = pd.read_csv("Boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)
ridge = Ridge(alpha=2.5)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print(r2_score(y_test, y_pred))

##### Grid Search CV
kfold = KFold(n_splits=5, shuffle=True, 
              random_state=2022)
ridge = Ridge()

params = {'alpha':np.linspace(0.001,11,20)}
gcv = GridSearchCV(ridge, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
print(best_model.coef_)

############# Concrete ########################
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")

concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

kfold = KFold(n_splits=5, shuffle=True, 
              random_state=2022)
ridge = Ridge()

params = {'alpha':np.linspace(0.001,11,20)}
gcv = GridSearchCV(ridge, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
print(best_model.coef_)