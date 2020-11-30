import pandas as pd
import numpy as np

# from sklearn.datasets import load_boston
# data = load_boston()
# # print(data)
#
# boston_data = pd.DataFrame(data.data, columns=data.feature_names)
# boston_data['Price'] = data.target
# print(boston_data.head())
# print(boston_data.shape)
# print(boston_data.dtypes)
# print(boston_data.info())
# print(boston_data.describe())

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
#
# # first do manually with train test split method
#
# X = boston_data.drop('Price', axis=1)
# y = boston_data.Price
# # print(X, y)
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit_transform(X)
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# regressor.fit(X_train, y_train)
#
# y_pred = regressor.predict(X_test)
#
# # print(y_test)
#
#
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# # print(df.head())
#
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, adjusted_rand_score
#
# # print(accuracy_score(X_train, y_train))
# # print(accuracy_score(y_test, y_pred))
# # print()
#
# # print(mean_squared_error(y_test, y_pred))
# # print(regressor.intercept_)
# # print(regressor.coef_)
#
# from sklearn.model_selection import cross_val_score, GridSearchCV
# score = cross_val_score(regressor, X, y, scoring='neg_mean_squared_error', cv=5)
# print(score.mean())
#
# print(r2_score(y_test, y_pred))
# print(adjusted_rand_score(y_test, y_pred))
# print()
# # Ridge regression to reduce the error here we have a lambda multiplied with slope to reduce the slope value
#
# from sklearn.linear_model import Ridge
# ridge = Ridge()
#
# param = {'alpha': [1, 2, 4, 5, 6, 10, 20, 40, 50, 60, 80, 100]}
#
# ridge_regressor = GridSearchCV(ridge, param_grid=param, scoring='neg_mean_squared_error', cv=5)
# ridge_regressor.fit(X, y)
# print(ridge_regressor.best_params_)
# print(ridge_regressor.best_score_)
#
# # Lasso regression to minimize the error and also helps in feature selections
#
# from sklearn.linear_model import Lasso
# lasso = Lasso()
#
# param = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 5, 6, 10, 20, 40, 50, 60, 80, 100]}
#
# lasso_regressor = GridSearchCV(lasso, param_grid=param, scoring='neg_mean_squared_error', cv=5)
# lasso_regressor.fit(X, y)
#
# print(lasso_regressor.best_params_)
# print(lasso_regressor.best_score_)

# Logistic Regression
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# print(data)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
# print(df.head())

# print(df.shape)
# print(df.dtypes)
# print(df.info())
# print(df.describe())

# from sklearn.linear_model import LogisticRegression
# log_regressor = LogisticRegression()

# If we have multiple class prediction then we have to use multi_class = 'ovr' with random state=0
#
X = df.drop('Target', axis=1)
y = df.Target

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# log_regressor.fit(X_train, y_train)
# y_pred = log_regressor.predict(X_test)
#
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#
# # print(confusion_matrix(y_test, y_pred))
# # print(classification_report(y_test, y_pred))
# # print(accuracy_score(y_test, y_pred))
#
# score = cross_val_score(log_regressor, X, y, scoring='accuracy', cv=5)
# print(score.mean())

# from sklearn.tree import DecisionTreeClassifier
# dec_clf = DecisionTreeClassifier()
#
# dec_clf.fit(X_train, y_train)
#
# y_pred = dec_clf.predict(X_test)
#
# # print(confusion_matrix(y_test, y_pred))
# # print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
#
# score = cross_val_score(dec_clf, X, y, scoring='accuracy', cv=5)
# print(score.mean())
#
# parameter = {'max_depth': range(3, 20)}
#
# dec_clf_Grid = GridSearchCV(dec_clf, param_grid=parameter, scoring='accuracy', cv=5)
# dec_clf_Grid.fit(X, y)
# print(dec_clf_Grid.best_params_)
# print(dec_clf_Grid.best_score_)

# KNN classifier
# from sklearn.neighbors import KNeighborsClassifier
# knn_clf = KNeighborsClassifier(n_neighbors=15)
#
# knn_clf.fit(X_train, y_train)
# y_pred = knn_clf.predict(X_test)

# print(accuracy_score(y_test, y_pred))


# score = cross_val_score(knn_clf, X, y, scoring='accuracy', cv=5)
# print(score.mean())

# parameter = {'n_neighbors': [1, 5, 10, 15, 20, 25, 50, 70, 90, 100]}
#
# grid_score = GridSearchCV(knn_clf, param_grid=parameter, scoring='accuracy', n_jobs=-1, verbose=3, cv=5)
# grid_score.fit(X, y)
# print(grid_score.best_params_)
# print(grid_score.best_score_)

# lets check the best k value

# k_range = range(1, 25)

# Accuracy = []
#
# for k in k_range:
#     knn_clf = KNeighborsClassifier(n_neighbors=k)
#     knn_clf.fit(X_train, y_train)
#     y_pred = knn_clf.predict(X_test)
#     Accuracy.append(accuracy_score(y_test, y_pred))

# Error = []
#
# for k in k_range:
#     knn_clf = KNeighborsClassifier(n_neighbors=k)
#     knn_clf.fit(X_train, y_train)
#     y_pred = knn_clf.predict(X_test)
#     Error.append(np.mean(y_pred != y_test))
#
# print(Error)

# print(Accuracy)

# now lets plot

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.lineplot(x=k_range, y= Accuracy)
# plt.xticks()
# plt.grid()
# plt.show()

# sns.lineplot(x=k_range, y=Error, marker='o', linestyle='--', markerfacecolor='red', markersize=5)
# plt.xticks()
# plt.grid()
# plt.show()

# Ensemble techniques ----- bagging (Random Forest)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

# predictions = cross_val_predict(rf_clf, X_test, y_test, cv=5)
# print(predictions)
# print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# print(accuracy_score(y_test, predictions))

# print(accuracy_score(y_test, y_pred))

# score = cross_val_score(rf_clf, X, y, scoring='accuracy', cv=5)
# print(score.mean())

# param = {'n_estimators': [100, 200, 400, 600],
#              'max_depth': [80, 90, 100, 110]}

# grid_score = GridSearchCV(rf_clf, param, scoring='recall_macro', cv=5)
# grid_score.fit(X, y)

# print(grid_score.best_params_)
# print(grid_score.best_score_)

from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
# randomised_score = RandomizedSearchCV(rf_clf, param_distributions=param, scoring='recall_macro', n_iter=10, n_jobs=-1, verbose=3, cv=5)
# randomised_score.fit(X, y)
# print(randomised_score.best_params_)
# print(randomised_score.best_score_)

# rf_clf = RandomForestClassifier(max_depth=80, n_estimators=200)
#
# score = cross_val_score(rf_clf, X, y, scoring='accuracy', cv=5)
# print(score)
# print(score.mean())


# How to handle imbalance dataset
data = {'Age': [27, 28, 25, 29, 26, 30, 31, 27, 28, 25, 29, 26, 30, 31], 'Experience': [3, 4, 2, 5, 2, 6, 7, 3, 4, 2, 5, 2, 6, 7],
        'Selection': [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]}
        # 'Selection': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']}
dataset = pd.DataFrame(data)
# print(dataset)
X = dataset.iloc[:, 0:2]
y = dataset.iloc[:, 2]
# print(dataset)

# undersampling ------ it is used where we have huge amount of data where the one label is less

# print(dataset.Selection.value_counts())
# print(X.shape)
# from imblearn.under_sampling import NearMiss
# nearmiss = NearMiss()
# X_new, y_new = nearmiss.fit_sample(X, y)
# # print(dataset.Selection.value_counts())
# # print(X_new.shape)
#
# from collections import Counter
#
# print(Counter(y))
# print(Counter(y_new))

# OverSampling

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
smote = SMOTETomek()

X_new, y_new = smote.fit_sample(X, y)

from collections import Counter

print(Counter(y))
print(Counter(y_new))