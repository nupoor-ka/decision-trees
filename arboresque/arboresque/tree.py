from ._metric import gini, entropy, mae, mse, friedman, poisson_deviance
from ._tree import build_tree, Tree
import numpy as np

class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        if criterion == "gini":
            self.criterion_func = gini
        elif criterion in ["entropy", "log_loss"]:
            self.criterion_func = entropy
        else:
            raise ValueError("unknown criterion")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True) # to prevent negative labels, labelling from 
        root = build_tree(
            X, y_encoded,
            criterion=self.criterion_func,
            leaf_val_func=None, # not needed for classification
            task="classification",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )
        self.tree_ = Tree(root=root)
        n_samples, n_features = X.shape
        self.n_features = n_features
        return self
    
    def _predict(self, x): # for one sample
        x = np.asarray(x)
        rt = self.tree_.root
        while rt:
            if rt.is_leaf:
                return self.classes_[rt.value]
            if x[rt.feature_index]<=rt.threshold:
                rt = rt.left
            else:
                rt = rt.right
    
    def predict(self, X): # for multiple samples at once
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_features = X.shape[1]
        if n_features != self.n_features:
            raise ValueError("Incorrect number of features")
        return np.array([self._predict(x) for x in X])

    def _predict_proba(self, x):
        x = np.asarray(x)
        rt = self.tree_.root
        while rt:
            if rt.is_leaf:
                return rt.class_counts/rt.n_samples
            if x[rt.feature_index]<=rt.threshold:
                rt = rt.left
            else:
                rt = rt.right

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_features = X.shape[1]
        if n_features != self.n_features:
            raise ValueError("Incorrect number of features")
        return np.array([self._predict_proba(x) for x in X])
    
    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

class DecisionTreeRegressor:
    def __init__(self, criterion="mse", max_depth=None, min_samples_split=2):
        self.criterion=criterion
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.leaf_val_func = np.mean
        if criterion == "mae":
            self.criterion_func = mae
            self.leaf_val_func = np.median
        elif criterion=="mse":
            self.criterion_func = mse
        elif criterion=="poisson":
            self.criterion_func = poisson_deviance
        elif criterion=="friedman":
            self.criterion_func = friedman
        else:
            raise ValueError("Unknown criterion")
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features = n_features
        root = build_tree(
            X, y,
            criterion=self.criterion_func,
            leaf_val_func=self.leaf_val_func,
            task="regression",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )
        self.tree_ = Tree(root=root)
        return self
    
    def _predict(self, x):
        x = np.asarray(x)
        rt = self.tree_.root
        while rt:
            if rt.is_leaf:
                return rt.value
            if x[rt.feature_index]<=rt.threshold:
                rt = rt.left
            else:
                rt = rt.right
    
    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_features = X.shape[1]
        if n_features != self.n_features:
            raise ValueError("Incorrect number of features")
        return np.array([self._predict(x) for x in X])
    
    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_pred = self.predict(X)
        v = np.sum((y - np.mean(y))**2)
        u = np.sum((y - y_pred)**2)
        if v==0:
            return 1
        return 1 - (u/v)
