import csv
import random

def load_train_test_datasets(filename, split=0.5):
    """Retrieve the training and test randomly split of the dataset
    based on the given split value

    :param filename: full path to the file containing the data
    :param split: the value between 0 and 1 (inclusive) used to split the data
    :return: a tuple with train and test sets of instances in the dataset
    """
    training_set = []
    test_set = []
    dataset = load_dataset(filename)
    for line in range(len(dataset)):
        if random.random() < split:
            training_set.append(dataset[line])
        else:
            test_set.append(dataset[line])
    return training_set, test_set


def load_dataset(filename):
    """Loads the dataset from a file

    :param filename: full path to the file containing the data
    :return: a list of instances in the dataset
    """
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(filter(None, lines))
        ncol = len(dataset[0])
        nlin = len(dataset)
        for line in range(nlin):
            for col in range(ncol - 1):
                dataset[line][col] = float(dataset[line][col])
    return dataset


toy_labeled_dataset = [[2.771244718, 1.784783929, 0],
                       [1.728571309, 1.169761413, 0],
                       [3.678319846, 2.81281357, 0],
                       [3.961043357, 2.61995032, 0],
                       [2.999208922, 2.209014212, 0],
                       [7.497545867, 3.162953546, 1],
                       [9.00220326, 3.339047188, 1],
                       [7.444542326, 0.476683375, 1],
                       [10.12493903, 3.234550982, 1],
                       [6.642287351, 3.319983761, 1]]

toy_unlabeled_dataset = [[2.771244718, 1.784783929],
                         [1.728571309, 1.169761413],
                         [3.678319846, 2.81281357],
                         [3.961043357, 2.61995032],
                         [2.999208922, 2.209014212],
                         [7.497545867, 3.162953546],
                         [9.00220326, 3.339047188],
                         [7.444542326, 0.476683375],
                         [10.12493903, 3.234550982],
                         [6.642287351, 3.319983761]]
