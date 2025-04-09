import numpy as np
import math
import pandas as pd
from collections import Counter

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    # make copy of lists first!
    X_copy = X.copy()
    y_copy = y.copy()

    # set random seed
    rand = np.random.default_rng(seed=random_state)

    # shuffle copy of lists
    if shuffle:
        indices = rand.permutation(len(X_copy))
        X_copy = [X_copy[i] for i in indices]
        y_copy = [y_copy[i] for i in indices]


    # calculate test size in int
    if test_size < 1:
        test_size = int(len(X_copy) * test_size + 0.9999)

    X_test = X_copy[-test_size:]
    y_test = y_copy[-test_size:]
    X_train = X_copy[:-test_size]
    y_train = y_copy[:-test_size]

    return X_train, X_test, y_train, y_test

def compute_euclidean_distance(x, y):
    """Computes the Euclidean distance between x and y.
    altered for categorical vals

    Args:
        x (list of int or float): first value
        y (list of int or float): second value

    Returns:
        dists: The Euclidean distance between vectors x and y.     
    """
    x = np.array(x, dtype=float)  # Convert to array
    y = np.array(y, dtype=float)
    return np.sqrt(np.sum((x - y) ** 2))

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct_count = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    if normalize:
        score = correct_count / len(y_true) if y_true else 0.0
    else:
        score = correct_count

    return score

def calculate_shannon_entropy(values):
    """
    Accepts a list of values and returns the entropy value. 
    H(X) = - Σ p(x) log(p(x))
    """
    total = len(values)
    counts = Counter(values)
    entropy = 0.0
    
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy

def calculate_renyi_entropy(series, alpha=2):
    """
    Accepts a list of values and returns the entropy value.
    """
    series = pd.Series(series)
    probabilities = series.value_counts(normalize=True)
    if alpha == 1:
        return -np.sum(probabilities * np.log2(probabilities))
    else:
        return 1 / (1 - alpha) * np.log2(np.sum(probabilities ** alpha))
