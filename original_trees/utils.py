"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame, columns = None) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    #we need to specify which columns need to be one-hot coded

    return pd.get_dummies(X, columns=columns, dtype=int)



def check_type(X: pd.DataFrame, y: pd.Series) -> str:
    """
    Function to check if the given series has real or discrete values
    """

    #inuitive approach, underoot of length < unique => real
    input_ = ""
    output_=""

    if np.sqrt(y.size) < y.unique().size:
        output_ = "r"
    else:
        output_ = "d"

    #assuming that the if discrete/real, every column will be discrete/real
    if np.sqrt(X.iloc[:,0].size) < X.iloc[:,0].unique().size:
        input_ = "r"
    else:
        input_ = "d"
    
    return input_+output_



def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    classes = Y.unique()
    counts_dict = Y.value_counts().to_dict()
    length = Y.size

    p = np.zeros(len(classes))
    entropy=0.0

    for i in range(len(classes)):
        p[i]=counts_dict[classes[i]]/length
        entropy+=-p[i]*np.log2(p[i])

    return entropy        


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    classes = Y.unique()
    counts_dict = Y.value_counts().to_dict()
    length = Y.size

    p = np.zeros(len(classes))
    sum=0.0

    for i in range(len(classes)):
        p[i]=counts_dict[classes[i]]/length
        sum+=p[i]**2

    return 1-sum

def mse(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    mean = np.mean(Y)
    
    return np.sum((mean - Y)**2)/Y.size 

def squared_error(Y: pd.Series)-> float:
    mean_y = np.mean(Y)
    return np.sum((mean_y -Y)**2)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str, case_: str):
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    
    Text: "gini_index", "entropy", "mse", "squared_error"

    return: float for discrete input, dataframe for real input
    """
    #FIX FOR REAL INPUT OR attr.dtype = float
    Y_t = Y
    

    if case_[0]=="d":
        #discrete input    => both cases 
        
        
        if criterion == "gini_index":
            fn = gini_index
        
        if criterion =="mse":
            fn = mse
        if criterion == "entropy":
            fn = entropy

        info_gain = fn(Y)
        reduction=0.0
        df = pd.DataFrame({"value":Y,"attribute":attr})

        for att_val in attr.unique():
            df_filtered = df[df["attribute"]==att_val]
            weight = len(df_filtered)/len(df)
            reduction+= weight*fn(df_filtered["value"])
        
        info_gain -= reduction
        Y = Y_t
        return info_gain
    else:
        #real input
        #step 1: I need to sort my table based on attributes,
        sorted_indices = attr.sort_values().index

        attr_sorted = attr.loc[sorted_indices]
        Y_sorted = Y.loc[sorted_indices]

        if criterion == "entropy":
            fn = entropy
        
        if criterion == "gini_index":
            fn = gini_index
        
        if criterion =="squared_error":
            fn = squared_error
        
        #Real input, Discrete output (Entropy and Gini are relevant here)
        info_gain = np.array([fn(Y_sorted)]*(Y.size-1))
        # print("info gain array before entropy weighting")
        # print(info_gain)

        if case_[1]=="d":
            #potential split number is one less than number of rows 
            

            for i in range(info_gain.shape[0]):
                lhs_entropy = fn(Y_sorted[:i+1]) #could be gini as well
                rhs_entropy = fn(Y_sorted[i+1:])

                weighted_entropy = (Y_sorted[:i+1].size/Y_sorted.size)*lhs_entropy + (Y_sorted[i+1:].size/Y_sorted.size)*rhs_entropy

                info_gain[i] -= weighted_entropy
            # print("info gain after weight")
            # print(
            # info_gain)
            # print("shape of info_gain",info_gain.shape[0])
            attr_sorted_half=(np.array(attr_sorted[0:attr_sorted.size-1]) + np.array(attr_sorted[1:attr_sorted.size])) / 2 #(taking midpoints for split)
            Y = Y_t
            return pd.DataFrame({"Split values":attr_sorted_half,"information_gain":info_gain})
        
        #Real input, Real output 
        else:
            loss_across = np.zeros(Y.size-1)
            
            for i in range(loss_across.shape[0]):
                lhs_se = fn(Y_sorted[:i+1]) #squared error is the function used 
                rhs_se = fn(Y_sorted[i+1:])

                loss_across[i] = lhs_se + rhs_se #we will pick the min loss to split 
            
            attr_sorted_half=(np.array(attr_sorted[0:attr_sorted.size-1]) + np.array(attr_sorted[1:attr_sorted.size])) / 2 #(taking midpoints for split)

            Y = Y_t
            return pd.DataFrame({"Split_values":np.array(attr_sorted_half),"loss":loss_across})
    


def opt_split_attribute_discrete_input(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series, case_:str):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).


    number_of_attributes = features.size 

    info_gain_arr = np.zeros(number_of_attributes)

    for i in range(number_of_attributes):
        info_gain_arr[i]=information_gain(y, X[features[i]], criterion,case_)

    index_max = np.argmax(info_gain_arr)

    return features[index_max]
    

def opt_split_attribute_real_input(X: pd.DataFrame, y: pd.Series, features: pd.Series, criterion, case_:str):
    """
    Function to find the optimal attribute to split about.
    This function works for all the cases with real input.
    
    features: pd.Series is a list of all the attributes we have to split upon (assuming all features are real)

    return: attribute to split upon
    """

    number_of_attributes = features.size 

    if case_[1]=="d":
        #real input, discrete output 
        df_final = pd.DataFrame(columns=["attribute","split_point","sel_or_infogain"])

        for i in range(number_of_attributes):
            feature = features[i]
            df = pd.DataFrame({"Label": y, "Attribute":X[feature]})

            df_returned =information_gain(df["Label"],df["Attribute"], criterion , case_) #what i recieved pd.DataFrame({"Features":features,"information_Gain":info_gain_arr}).sort_values(by="Information Gain")
            # print("this is df returend")
            df_returned = df_returned.sort_values(by="information_gain", ascending=False) 
            df_returned.reset_index(drop=True, inplace=True)
            # print(df_returned.head(5))
            df_final.loc[i]= [feature,df_returned["Split values"][0],df_returned["information_gain"][0]]
            # print("this is the final df")
            # print('df_final', [feature,df_returned["Split values"][0],df_returned["information_gain"][0]])
        # print("#"*20)
        # print("this is the final df")
        # print(df_final) #######
        # print(y)
        return df_final
    else:
        #Real input, real output
        df_final = pd.DataFrame(columns=["attribute","split_point","sel_or_infogain"])

        for i in range(number_of_attributes):
            feature = features[i]
            df = pd.DataFrame({"Label": y, "Attribute":X[feature]})

            df_returned =information_gain(df["Label"],df["Attribute"], "squared_error",case_) #pd.DataFrame({"Split Values":np.array(attr_sorted_half),"loss":loss_across})
            df_returned = df_returned.sort_values(by="loss", ascending=True) #the most minimum loss is what is best
            df_returned.reset_index(drop=True, inplace=True)


            df_final.loc[i]=[feature,df_returned["Split_values"][0],df_returned["loss"][0]]
        
        return df_final



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value, case_: str):
    """
    Function to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real-valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon
    case_: whether the attribute is discrete ('d') or real-valued ('r')

    return: split data (Input and output)
    """
    
    # Create a copy to avoid modifying the original DataFrame
    X_copy = X.copy()
    X_copy["y_label"] = y

    if case_[0] == "d":
        # Discrete input will always have the same split
        df_right = X_copy[X_copy[attribute] == 1]  # Assuming one-hot encoding for discrete
        df_left = X_copy[X_copy[attribute] == 0]

        y_right = df_right["y_label"]
        y_left = df_left["y_label"]

        df_right = df_right.drop("y_label", axis=1)
        df_left = df_left.drop("y_label", axis=1)

        return (df_left, y_left), (df_right, y_right)

    elif case_[0] == "r":
        # Ensure 'value' is a scalar
        if not pd.api.types.is_scalar(value):
            raise ValueError("The 'value' parameter should be a scalar for real-valued splits.")
        
        # Handle the real-valued input split
        df_right = X_copy[X_copy[attribute] >= value]
        df_left = X_copy[X_copy[attribute] < value]

        y_right = df_right["y_label"]
        y_left = df_left["y_label"]

        df_right = df_right.drop("y_label", axis=1)
        df_left = df_left.drop("y_label", axis=1)

        return (df_left, y_left), (df_right, y_right)

    else:
        raise ValueError("Invalid case_ value. Use 'd' for discrete and 'r' for real-valued attributes.")

def predict_helper(tree_helper,case_, branch_label_helper, row):
    """
    Function to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real-valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon
    case_: whether the attribute is discrete ('d') or real-valued ('r')

    return: split data (Input and output)
    """
    #print(tree[branch_label_helper])

    data_type =  type(tree_helper[branch_label_helper])

    if  data_type != dict:
        # print(tree_helper[branch_label_helper])
        return_value = tree_helper[branch_label_helper].copy()
        return return_value
    

    # print("check this out", branch_label_helper)
    # print(type(tree_helper[branch_label_helper]))
     
    if case_[0]=="d":
        #discrete input
        val_att = row[tree_helper[branch_label_helper]["attribute"]]
        if val_att==1:
            branch_label_helper=tree_helper[branch_label_helper]["right_label"]
        else:
            branch_label_helper=tree_helper[branch_label_helper]["left_label"]
        
        return predict_helper(tree_helper,case_, branch_label_helper, row)

    else:
        #real input 
        val_att = row[tree_helper[branch_label_helper]["attribute"]]
        split_value = tree_helper[branch_label_helper]["split_value"]

        if val_att<split_value:
            branch_label_helper=tree_helper[branch_label_helper]["right_label"]
        else:
            branch_label_helper=tree_helper[branch_label_helper]["left_label"]
        
        return predict_helper(tree_helper,case_, branch_label_helper, row)