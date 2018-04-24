from python import data_helper
from python import distance
from python import evaluation
import operator


def find_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = distance.euclidean(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def classify(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes, key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0]


def main():
    split = 0.67
    k = 3
    dataset_path = r'../dataset/iris.data'

    # prepare data
    training_set, test_set = data_helper.load_dataset(dataset_path, split)
    # print('Train set: ' + repr(len(training_set)))
    # print('Test set: ' + repr(len(test_set)))

    # generate predictions
    predictions = []
    # print(' Predicted \t\t| Actual\n')

    for x in range(len(test_set)):
        neighbors = find_neighbors(training_set, test_set[x], k)
        predictions.append(classify(neighbors))
        # print('> {} \t| {}'.format(repr(predictions[-1]), repr(test_set[x][-1])))

    accuracy = evaluation.accuracy(test_set, predictions)
    print('Accuracy is {0:.2f}'.format(accuracy))


if __name__ == '__main__':
    main()
