from ._metric import gini, entropy, mae, mse, friedman, poisson_deviance
from ._tree import build_tree, Tree
import numpy as np
import warnings

class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 categorical_features=None, min_samples_leaf=1, max_features=None,
                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0):
        """
        yet to add
        min_weight_fraction_leaf
        """
        if not isinstance(criterion, str):
            raise ValueError("criterion should be a string from {'gini', 'entropy', 'log_loss'}.")
        if criterion == "gini":
            self.criterion_func = gini
        elif criterion in ["entropy", "log_loss"]:
            self.criterion_func = entropy
        else:
            raise ValueError("Unknown criterion.")
        self.criterion = criterion

        if not isinstance(min_samples_split, (float, int)):
            raise ValueError("min_samples_split should be an int or float.")
        self.min_samples_split = min_samples_split

        if not isinstance(min_samples_leaf, (int, float)):
            raise ValueError("min_samples_leaf should be an int or float.")
        self.min_samples_leaf = min_samples_leaf
        
        if categorical_features is not None and not isinstance(categorical_features, list):
            raise ValueError("categorical_features should be a list of non-negative indices.")
        self.cat_fts=categorical_features

        if max_depth is not None and not isinstance(max_depth, int):
            raise ValueError("max_depth should be a non-negative int.")
        self.max_depth = max_depth

        if max_features is not None and not isinstance(max_features, (int, float, str)):
            raise ValueError("max_features should be an int, a float or a string from {'sqrt', 'log2'}")
        self.max_features = max_features

        if random_state is not None and not isinstance(random_state, int):
            raise ValueError("random_state should be an int.")
        self.random_state = random_state
        
        if max_leaf_nodes is not None and not isinstance(max_leaf_nodes, int):
            raise ValueError("max_leaf_nodes should be an int.")
        self.max_leaf_nodes = max_leaf_nodes

        if not isinstance(min_impurity_decrease, (float,int)):
            raise ValueError("min_impurity_decrease should be a float.")
        self.min_impurity_decrease = min_impurity_decrease
        
    def _handle_cat_fts_fit(self, X):
        if len(self.cat_fts)!=len(set(self.cat_fts)):
            raise ValueError("categorical_features contains duplicate entries.")
        n_samples, n_features = X.shape
        index_cols = {} # index to insert at, column to add, numpy series, insert axis 1
        self.cat_ft_info = {}
        for ft in self.cat_fts:
            if ft<0:
                raise ValueError(f"Negative indices in categorical_features are not supported. Got {ft}.")
            if ft>=n_features:
                raise ValueError(f"The index {ft} exceeds valid indices, "
                                    f"the number of features is {n_features}, "
                                    f"highest valid index is {n_features-1}.")
            cats = np.unique(X[:, ft])
            num_cats = len(cats)
            if num_cats>10:
                msg = (f"The feature at index {ft} has {num_cats} categories. "
                       f"The current version of this library can only handle "
                       f"features with at most ten categories as categorical features.")
                warnings.warn(msg)
                continue
            new_cols = np.zeros((n_samples,num_cats))
            cat_to_index = {}
            for cat_i, cat in enumerate(cats):
                cat_to_index[cat] = cat_i
            self.cat_ft_info[ft] = cat_to_index
            for num, sample in enumerate(X[:, ft]):
                new_cols[num, cat_to_index[sample]]=1
            index_cols[ft] = new_cols
        xnew_cols = []
        for col_indx in range(n_features):
            if col_indx in self.cat_ft_info:
                xnew_cols.append(index_cols[col_indx])
            else:
                xnew_cols.append(X[:,col_indx:col_indx+1])
        X_new = np.hstack(xnew_cols)
        return X_new
    
    def _handle_cat_fts_pred(self, X):
        index_cols = {}
        xrows, xcols = X.shape 
        for ft, cat_to_index in self.cat_ft_info.items():
            new_cols =  np.zeros((xrows, len(cat_to_index)))
            for num, sample in enumerate(X[:, ft]):
                if sample in cat_to_index: # else haven't seen this category
                    new_cols[num, cat_to_index[sample]]=1
            index_cols[ft] = new_cols
        xnew_cols = []
        for col_indx in range(xcols):
            if col_indx in self.cat_ft_info:
                xnew_cols.append(index_cols[col_indx])
            else:
                xnew_cols.append(X[:,col_indx:col_indx+1])
        X_new = np.hstack(xnew_cols)
        return X_new

    def fit(self, X, y):
        """
        yet to add class weights and sample weights
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True) # to prevent negative labels, labelling from 
        self.n_classes = len(self.classes_)

        if self.cat_fts is not None:
            X_prepped = self._handle_cat_fts_fit(X)
        else:
            X_prepped = X

        n_samples, n_features = X_prepped.shape
        self.n_features = n_features

        if isinstance(self.min_samples_split, float):
            self.min_samples_split = np.ceil(self.min_samples_split * n_samples)

        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf = np.ceil(self.min_samples_leaf * n_samples)

        if self.max_features is not None:
            if isinstance(self.max_features, int):
                if self.max_features<=0:
                    raise ValueError("max_features should be greater than 0.")
                self.max_features=min(n_features, self.max_features)
            elif isinstance(self.max_features, float):
                self.max_features = max(1, int(self.max_features * n_features))
            elif isinstance(self.max_features, str):
                if self.max_features=="sqrt":
                    self.max_features = int(np.sqrt(n_features))
                elif self.max_features=="log2":
                    self.max_features = max(1, int(np.log2(n_features)))

        if self.random_state is not None:
            np.random.seed(self.random_state)

        root = build_tree(
            X_prepped, y_encoded,
            criterion=self.criterion_func,
            leaf_val_func=None, # not needed for classification
            task="classification",
            n_classes=self.n_classes,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            n_total_samples=n_samples,
        )
        self.tree_ = Tree(root=root)
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
        if self.cat_fts:
            X_prepped = self._handle_cat_fts_pred(X)
        else:
            X_prepped = X
        n_features = X_prepped.shape[1]
        if n_features != self.n_features:
            raise ValueError("Incorrect number of features")
        return np.array([self._predict(x) for x in X_prepped])

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
        if self.cat_fts:
            X_prepped = self._handle_cat_fts_pred(X)
        else:
            X_prepped = X
        n_features = X_prepped.shape[1]
        if n_features != self.n_features:
            raise ValueError("Incorrect number of features")
        return np.array([self._predict_proba(x) for x in X_prepped])
    
    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred==y)
    
    def get_depth(self):
        return self.tree_.get_depth()
    
    def get_n_leaves(self):
        return self.tree_.get_n_leaves()

class DecisionTreeRegressor:
    def __init__(self, criterion="mse", max_depth=None, min_samples_split=2):
        if not isinstance(criterion, str):
            ValueError("Criterion should be a string from {'mse', 'mae', 'poisson', 'friedman'}.")
        self.criterion=criterion
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
            raise ValueError("Unknown criterion.")

        if not isinstance(max_depth, int):
            raise ValueError("max_depth should be a non-negative int.")
        self.max_depth = max_depth

        self.min_samples_split=min_samples_split
    
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
    
    def get_depth(self):
        return self.tree_.get_depth()
    
    def get_n_leaves(self):
        return self.tree_.get_n_leaves()
