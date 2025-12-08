import numpy as np

def find_best_split(X, y, criterion):
    """
    criterion is the impurity metric user chose
    that criterion determines whether regression or classification
    best feature, best split point or threshold, information gain
    """
    n_samples, n_features = X.shape
    if n_samples <= 1: # ideally this should be checked in the build function
        return None, None, 0.0
    
    parent_impurity = criterion(y)
    
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx]) # possible thresholds for this feature
        
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0: # should only be the case for right
                continue
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            n_left = len(y_left)
            n_right = len(y_right)

            weighted_impurity = (n_left / n_samples) * criterion(y_left) + \
                                (n_right / n_samples) * criterion(y_right)
            
            gain = parent_impurity - weighted_impurity
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain
