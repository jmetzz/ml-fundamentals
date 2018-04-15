from python import data_helper, distance
import random
import numpy as np

"""Given the inputs x1,x2,x3,…,xn and value of K

    Step 1 - Pick K random points as cluster centers called centroids.
    Step 2 - Assign each xi_i to nearest cluster by calculating its distance to each centroid.
    Step 3 - Find new cluster center by taking the average of the assigned points.
    Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.
"""


def dist(instance, centroids, function_dist=distance.euclidean):
    ncol = len(instance)
    return [function_dist(instance, c, ncol) for c in centroids]


def _initial_centroids(k, data):
    indexes = set(random.sample(range(0, len(data)), k))
    return [data[i] for i in indexes]


def find_clusters(data, centroids, function_dist=distance.euclidean):
    """Distribute the instances in the clusters represented by the centroids

    :param data: the dataset of instances
    :param centroids: the centroid vectors with the same structure as the instances in the dataset
    :param function_dist: the function used to calculate the distance between two instances
    :return: a list representing the cluster assignment for each instance in the dataset
    """
    clusters = [0] * len(data)
    for idx in range(len(data)):
        distances = dist(data[idx], centroids)
        cluster = distances.index(min(distances))
        clusters[idx] = cluster
    return clusters


def calculate_centroids(data, k, clusters):
    centroids = [0] * k
    for c in range(k):
        points = [data[j] for j in range(len(data)) if clusters[j] == c]
        centroids[c] = np.mean(points, axis=0).tolist()
    return centroids


def centroids_changed(prev_centroids, new_centroids):
    c1 = [item for sublist in prev_centroids for item in sublist]
    c2 = [item for sublist in new_centroids for item in sublist]
    return c1 != c2


def main():
    data = data_helper.toy_dataset()
    k = 3
    prev_centroids = _initial_centroids(k, data)
    new_centroids = [[-1] * len(data[0])] * k

    # TODO - stop condition of the main loop
    while centroids_changed(prev_centroids, new_centroids):
        clusters = find_clusters(data, prev_centroids)
        new_centroids = calculate_centroids(data, k, clusters)

    print(new_centroids)


if __name__ == '__main__':
    main()





# def _closest_centroid(idx, data, centroids, function_dist=distance.euclidean):
#     """Gets the index of the closest centroid to the given instance
#
#     :param idx: the instance index
#     :param data: the dataset of instances
#     :param centroids: the centroid vectors with the same structure as the instances in the dataset
#     :return: the index of the closest centroid
#     """
#     dist = []
#     ncol = len(data[0])
#     for c in centroids:
#         dist.append(function_dist(data[idx], c, ncol))
#     return dist.index(min(dist))

# def calculate_centroids(data, map):
#     centroids = []
#     for entry in map.items():
#         inst_indexes = entry.value
#         cluster_size = len(inst_indexes)
#         cluster = [data[i] for i in inst_indexes]
#         centroids.append()
