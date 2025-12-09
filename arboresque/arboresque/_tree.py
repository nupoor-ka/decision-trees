import numpy as np
from ._split import find_best_split

"""
will define tree, node and some funcs here
fit, predict, predict_proba, apply, decision_path, get_depth, get_n_leaves
"""

class Node:
    def __init__(self,
                 feature_index=None,
                 threshold=None,
                 left=None,
                 right=None,
                 value=None,
                 n_samples=0,
                 class_counts=None):
        self.feature_index = feature_index # column to split on
        self.threshold = threshold # split point
        self.left = left # Node or None
        self.right = right
        self.value = value # class dist or reg value for this leaf
        self.n_samples = n_samples # samples at this node
        self.is_leaf = (left is None and right is None)
        self.class_counts = class_counts # None for regression

class Tree:
    def __init__(self, root=None):
        self.root = root
    def get_depth(self):
        q = [self.root]
        depth = 0
        while q:
            l = len(q)
            for i in range(l):
                rt = q[i]
                q.pop(0)
                if rt.left:
                    q.append(rt.left)
                if rt.right:
                    q.append(rt.right)
            depth+=1
        return depth
    def get_n_leaves(self):
        q = [self.root]
        n_leaves = 0
        while q:
            l = len(q)
            for i in range(l):
                rt = q[i]
                if rt.is_leaf:
                    n_leaves+=1
                    continue
                q.pop(0)
                if rt.left:
                    q.append(rt.left)
                if rt.right:
                    q.append(rt.right)
        return n_leaves

def build_tree(X, y, criterion, leaf_val_func, task, depth=0, max_depth=None, min_samples_split=2):
    """
    max_depth and min_samples_split are stopping options
    max_depth : int or None
    min_samples_split : int
    output is root of tree
    recursive function
    """
    n_samples = len(y)
    
    pure_node = False
    cc = None
    if task=='classification':
        unique_classes, counts = np.unique(y, return_counts=True)
        value = unique_classes[np.argmax(counts)] # majority class
        if len(unique_classes) == 1:
            pure_node=True
        cc = counts
    else:
        value = leaf_val_func(y)
    
    # stopping conditions
    if (max_depth is not None and depth >= max_depth) or \
       (n_samples < min_samples_split) or \
       pure_node: # pure node
        return Node(value=value, n_samples=n_samples)
    
    # best split
    best_feature, best_threshold, best_gain = find_best_split(X, y, criterion)
    
    # no valid split found
    if best_feature is None or best_gain == 0:
        return Node(value=value, n_samples=n_samples)
    
    # split
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    
    # recurse
    left_child = build_tree(X_left, y_left, criterion, leaf_val_func, task, depth + 1, max_depth, min_samples_split)
    right_child = build_tree(X_right, y_right, criterion, leaf_val_func, task, depth + 1, max_depth, min_samples_split)

    return Node(
        feature_index=best_feature,
        threshold=best_threshold,
        left=left_child,
        right=right_child,
        value=value, # backup value if needed
        n_samples=n_samples,
        class_counts=cc
    )
