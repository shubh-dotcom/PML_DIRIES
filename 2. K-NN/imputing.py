import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

job = pd.read_csv("JobSalary2.csv")
## Finding NAs in columns
job.isnull().sum()

# dropping the rows with NA values
job.dropna()

# Constant Imputer
imp = SimpleImputer(strategy='constant',fill_value=50)
imp.fit_transform(job)

job.mean()
# Mean Imputer
imp = SimpleImputer(strategy='mean')
imp.fit_transform(job)

# Median Imputer
imp = SimpleImputer(strategy='median')
np_imp = imp.fit_transform(job)

pd_imp = pd.DataFrame(np_imp, 
                      columns=job.columns)

############ Chemical Process #############
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Chemical Process Data")

chemdata = pd.read_csv("ChemicalProcess.csv")
## Finding NAs in columns
chemdata.isnull().sum()

X = chemdata.drop("Yield", axis=1)
y = chemdata['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)
imp = SimpleImputer(strategy='median')
X_trn_trf = imp.fit_transform(X_train)
X_tst_trf = imp.transform(X_test)

lr = LinearRegression()
lr.fit(X_trn_trf, y_train)
y_pred = lr.predict(X_tst_trf)
print(r2_score(y_test, y_pred))

### with Pipeline
from sklearn.pipeline import Pipeline
imp = SimpleImputer(strategy='median')
lr = LinearRegression()
pipe = Pipeline([('IMPUTE',imp),('LR',lr)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))

########### K-NN ###########
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV

imp = SimpleImputer()
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('IMPUTE',imp),('STD',scaler),('KNN',knn)])
print(pipe.get_params())
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'IMPUTE__strategy':['mean','median'],
          'KNN__n_neighbors': np.arange(1,11)}
gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)













