import pprint as pp
import random

import python.data_helper as data_helper
import python.distance as distance

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


def mean(points):
    ncols = len(points[0])
    m = [0] * ncols
    for col in range(ncols):
        for p in points:
            m[col] += p[col]
        m[col] = m[col] / len(points)

    return m


def calculate_centroids(data, k, clusters):
    centroids = [0.0] * k
    for c in range(k):
        points = [data[j] for j in range(len(data)) if clusters[j] == c]
        centroids[c] = mean(points)
    return centroids


def centroids_changed(prev_centroids, new_centroids):
    for i in range(len(prev_centroids)):
        for z1, z2 in zip(prev_centroids[i], new_centroids[i]):
            if z1 != z2:
                return True
    return False


def report(data, new_centroids, clusters):
    print("\nCentroids:")
    print(new_centroids)
    print("clusters: [Clusters | instance]")
    map = {c: [] for c in set(clusters)}
    for e in range(len(data)):
        map.get(clusters[e]).append(data[e])
    pp.pprint(map)


def main():
    data = data_helper.toy_unlabeled_dataset
    k = 3
    clusters = [0] * len(data[0])
    prev_centroids = [[-1] * len(data[0])] * k
    new_centroids = _initial_centroids(k, data)
    while centroids_changed(prev_centroids, new_centroids):
        clusters = find_clusters(data, new_centroids)
        prev_centroids = new_centroids
        new_centroids = calculate_centroids(data, k, clusters)
        # print(new_centroids)

    report(data, new_centroids, clusters)

if __name__ == '__main__':
    main()

