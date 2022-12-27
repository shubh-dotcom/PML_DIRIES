import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score

mowers = pd.read_csv("RidingMowers.csv")

sns.scatterplot(data=mowers, x='Income',
                y='Lot_Size', hue='Response')
plt.show()


dum_mow = pd.get_dummies(mowers, drop_first=True)

X = dum_mow.drop('Response_Not Bought', axis=1)
y = dum_mow['Response_Not Bought']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)
print(y.value_counts(normalize=True)*100)
print(y_train.value_counts(normalize=True)*100)
print(y_test.value_counts(normalize=True)*100)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
# P(y == 1)
y_pred_prob = knn.predict_proba(X_test)[:,1]
prediction = np.where(y_pred_prob>0.5, 1,0)


print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Loop
acc = []
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))

i_max = np.argmax(acc)
best_k = Ks[i_max]
print("Best n_neigbors =", best_k)


####### ROC AUC ##################
comp_prob = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\7. ML Basics\comp_prob.csv")
## Evaluation of Model 1 Probs
fpr, tpr, thresholds = roc_curve(comp_prob['y_test'],
                                 comp_prob['yprob_1'])

plt.plot(fpr, tpr)
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
plt.show()

print(roc_auc_score(comp_prob['y_test'], 
                    comp_prob['yprob_1']))

## Evaluation of Model 2 Probs
fpr, tpr, thresholds = roc_curve(comp_prob['y_test'],
                                 comp_prob['yprob_2'])

plt.plot(fpr, tpr)
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
plt.show()

print(roc_auc_score(comp_prob['y_test'], 
                    comp_prob['yprob_2']))

################################################


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_prob = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test,
                                 y_pred_prob)

plt.plot(fpr, tpr)
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
plt.show()

print(roc_auc_score(y_test, y_pred_prob))

from sklearn.metrics import roc_curve, roc_auc_score
### loop
acc = []
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_prob = knn.predict_proba(X_test)[:,1]
    acc.append(roc_auc_score(y_test, y_pred_prob))

i_max = np.argmax(acc)
best_k = Ks[i_max]
print("Best n_neigbors =", best_k)


########### Log Loss ###########################
from sklearn.metrics import log_loss
comp_prob = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\7. ML Basics\comp_prob.csv")
## Evaluation of Model 1 Probs

print(log_loss(comp_prob['y_test'], 
                    comp_prob['yprob_1']))

## Evaluation of Model 2 Probs

print(log_loss(comp_prob['y_test'], 
                    comp_prob['yprob_2']))

from sklearn.metrics import log_loss
### loop
acc = []
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_prob = knn.predict_proba(X_test)[:,1]
    acc.append(-log_loss(y_test, y_pred_prob))

i_max = np.argmax(acc)
best_k = Ks[i_max]
print("Best n_neigbors =", best_k)
