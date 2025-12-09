import numpy as np
from arboresque import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import datasets

X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y_clf = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_clf = [-1, 1, 1]

y_reg = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
true_reg = [1.0, 3.0, 3.0]

def test_classification_toy():
    clf = DecisionTreeClassifier()
    clf.fit(X, y_clf)
    y_pred = clf.predict(T)
    assert np.array_equal(y_pred, true_clf)

def test_regression_toy():
    reg = DecisionTreeRegressor()
    reg.fit(X, y_reg)
    y_pred = reg.predict(T)
    assert np.allclose(y_pred, true_reg)
