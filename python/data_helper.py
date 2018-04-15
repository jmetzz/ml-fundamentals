import csv
import random

def load_dataset(filename, split=0.5):
    """Retrieve the training and test randomly split of the dataset
    based on the given split value

    :param filename: full path to the file containing the data
    :param split: the value between 0 and 1 (inclusive) used to split the data
    :return: a tuple with train and test sets of instances in the dataset
    """
    training_set = []
    test_set = []
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        ncol = len(dataset[0])
        nlin = len(dataset)
        for line in range(nlin):
            for col in range(ncol - 1):
                dataset[line][col] = float(dataset[line][col])
            if random.random() < split:
                training_set.append(dataset[line])
            else:
                test_set.append(dataset[line])
    return training_set, test_set


