import numpy as np

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
