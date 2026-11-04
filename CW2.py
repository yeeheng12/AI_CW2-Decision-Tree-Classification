# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Loke Yee Heng
# STUDENT EMAIL:  lnnx6136@leeds.ac.uk

def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)

# Task 1 [8 marks]: 
def load_data(file_path, delimiter=','):
    num_rows, data, header_list=None, None, None
    if not os.path.isfile(file_path):
        warnings.warn(f"CSV file '{file_path}' does not exist.")
        return None, None, None

    # Insert your code here for task 1
    # read csv file, convert data frame to Numpy array,get number list, extract header list
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        data = df.to_numpy()
        num_rows = df.shape[0]
        header_list = df.columns.tolist()
    except Exception as e:
        warnings.warn(f"Task 1: Error - {str(e)}")

    return num_rows, data, header_list

# Task 2[8 marks]:
def filter_data(data):
    filtered_data=[None]*1
    # Insert your code here for task 2
    # create Boolean mask for filtering
    # ~ set -99 value as True for filtering
    mask = ~np.any(data == -99, axis=1)
    filtered_data = data[mask]

    return filtered_data

# Task 3 [8 marks]:
def statistics_data(data):
    coefficient_of_variation=None
    # Insert your code here for task 3
    # filter data with task 2 function
    filtered_data = filter_data(data)
    # exclude label column by slicing, convert to float data type
    feature = filtered_data[:, :-1].astype(float)
    means = np.mean(feature, axis=0)
    std_devs = np.std(feature, axis=0)
    # final calculation, feature with zero mean return infinity
    coefficient_of_variation = np.where(means == 0, np.inf, std_devs / means)

    return coefficient_of_variation

# Task 4 [8 marks]:
def split_data(data, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test=None, None, None, None
    np.random.seed(1)
    # filter with task 2
    filtered_data = filter_data(data)
    # separate features(x) and labels(y), use same slicing approach as task 3
    x = filtered_data[:, :-1]
    y = filtered_data[:, -1]
    # use method from scikit-learn library
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

    return x_train, x_test, y_train, y_test

# Task 5 [8 marks]:
def train_decision_tree(x_train, y_train,ccp_alpha=0):
    model=None
    # Insert your code here for task 5
    # initialize and train model with given ccp_alpha
    model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    model.fit(x_train, y_train)

    return model

# Task 6 [8 marks]:
def make_predictions(model, X_test):
    y_test_predicted=None
    # Insert your code here for task 6
    y_test_predicted = model.predict(X_test)

    return y_test_predicted

# Task 7 [8 marks]:
def evaluate_model(model, x, y):
    accuracy, recall=None,None
    # Insert your code here for task 7
    # get predictions
    y_predict = model.predict(x)
    # Compute accuracy and recall
    accuracy = accuracy_score(y, y_predict)
    recall = recall_score(y, y_predict)

    return accuracy, recall

# Task 8 [8 marks]:
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    optimal_ccp=None
    # Insert your code here for task 8
    # train initial unpruned model with ccp_alpha=0
    unpruned_model = train_decision_tree(x_train, y_train, ccp_alpha=0)
    unpruned_accuracy, _ = evaluate_model(unpruned_model, x_test, y_test)

    # if unpruned model achieves perfect accuracy return 0
    if unpruned_accuracy == 1.0:
        return float(0.0)
    # define acceptable accuracy range
    lower_bound = unpruned_accuracy - 0.01
    upper_bound = unpruned_accuracy + 0.01

    # initialize search for optimal ccp_alpha
    last_valid_ccp_alpha = 0.0
    max_iterations = 1000  # prevent infinite loops
    step_size = 0.001
    current_ccp_alpha = step_size

    # run for max iterations
    for _ in range(max_iterations):
        # train model with current ccp_alpha
        model = train_decision_tree(x_train, y_train, ccp_alpha=current_ccp_alpha)
        accuracy, _ = evaluate_model(model, x_test, y_test)
        # check accuracy is within acceptable range
        if lower_bound <= accuracy <= upper_bound:
            last_valid_ccp_alpha = current_ccp_alpha
        else:
            break  # stop if accuracy drops below threshold

        current_ccp_alpha += step_size

        # stop if ccp_alpha reaches 1.0
        if current_ccp_alpha > 1.0:
            break

    # return the last valid ccp_alpha before accuracy drop
    optimal_ccp = last_valid_ccp_alpha

    return optimal_ccp

# Task 9 [8 marks]:
def tree_depths(model):
    depth=None
    # Get the depth of the unpruned tree
    # Insert your code here for task 9
    # get depth of decision tree
    depth = model.get_depth()

    return depth

 # Task 10 [8 marks]:
def important_feature(x_train, y_train,header_list):
    best_feature=None
    # Insert your code here for task 10
    # initialize variables
    last_non_zero_feature = None
    max_iterations = 100  # prevent infinite loop
    step_size = 0.01
    ccp_alpha = 0.0

    for _ in range(max_iterations):
        # train tree model with current ccp_alpha
        model = train_decision_tree(x_train, y_train, ccp_alpha=ccp_alpha)
        depth = tree_depths(model)

        # if depth is 1
        if depth == 1:
            # obtain result
            tree_rules = model.tree_
            best_feature_index = tree_rules.feature[0]
            best_feature = header_list[best_feature_index]
            return best_feature

        # if depth was previously greater than 1 but drop to 0, return last non-zero depth feature
        if depth == 0 and last_non_zero_feature is not None:
            return last_non_zero_feature

        # stores the last feature used
        if depth > 1:
            tree_rules = model.tree_
            best_feature_index = tree_rules.feature[0]  # First feature used in the tree
            last_non_zero_feature = header_list[best_feature_index]

        ccp_alpha += step_size

        # stop if ccp_alpha exceeds 1
        if ccp_alpha > 1.0:
            break

    return best_feature

# Task 11 [10 marks]:
def optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list):
    optimal_ccp=None
    # Insert your code here for task 11
    # identify the most important feature using task 10
    most_important_feature = important_feature(x_train, y_train, header_list)

    if most_important_feature is None:
        warnings.warn("Warning - Could not determine the most important feature.")
        return float(0.0)

    # get index
    feature_index = header_list.index(most_important_feature)

    # create new datasets only with this feature
    x_train_single = x_train[:, feature_index].reshape(-1, 1)
    x_test_single = x_test[:, feature_index].reshape(-1, 1)

    # identify optimal ccp_alpha using task 8 implementation
    optimal_ccp = optimal_ccp_alpha(x_train_single, y_train, x_test_single, y_test)

    # ensure return value is float data type
    if optimal_ccp is None:
        return float(0.0)

    return optimal_ccp

# Task 12 [10 marks]:
def optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list):
    optimal_depth=None
    # Insert your code here for task 12
    # find the most important feature using the method you created in task 10
    most_important_feature = important_feature(x_train, y_train, header_list)

    if most_important_feature is None:
        warnings.warn("Warning - Could not determine the most important feature in task 12.")
        return None

    # remove most important feature and find second most, create a reduced feature dataset excluding the first important feature
    feature_index_1 = header_list.index(most_important_feature)
    x_train_reduced = np.delete(x_train, feature_index_1, axis=1)
    reduced_header_list = header_list[:feature_index_1] + header_list[feature_index_1 + 1:]

    # find second most important feature
    second_most_important_feature = important_feature(x_train_reduced, y_train, reduced_header_list)

    if second_most_important_feature is None:
        warnings.warn("Warning - Could not determine the second most important feature.")
        return None

    # get index of both selected features, extract data with these features
    feature_index_2 = header_list.index(second_most_important_feature)
    x_train_two = x_train[:, [feature_index_1, feature_index_2]]
    x_test_two = x_test[:, [feature_index_1, feature_index_2]]

    # find optimal ccp_alpha with task 8, then train pruned tree with value
    optimal_ccp = optimal_ccp_alpha(x_train_two, y_train, x_test_two, y_test)
    model = train_decision_tree(x_train_two, y_train, ccp_alpha=optimal_ccp)

    # obtain depth
    optimal_depth = tree_depths(model)

    return optimal_depth

# Example usage (Main section):
if __name__ == "__main__":
    # Load data
    file_path = r"C:\Users\OWNER\AI_CW2\DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}");
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}");
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)

    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)

    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)

    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)

    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)

    # Test optimal ccp_alpha with single feature
    optimal_alpha_single = optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal ccp_alpha using single most important feature: {optimal_alpha_single:.4f}")
    print("-" * 50)

    # Test optimal depth with two features
    optimal_depth_two = optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal tree depth using two most important features: {optimal_depth_two}")
    print("-" * 50)

# References:
# Line 45-46 is inspired by a code at https://www.geeksforgeeks.org/how-to-filter-two-dimensional-numpy-array-based-on-condition
# Line 75 is inspired by a code at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split
# Line 84 is inspired by a code at https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier
# Line 93 is inspired by a code at https://scikit-learn.org/stable/glossary
# Line 158 is inspired by a code at https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier


