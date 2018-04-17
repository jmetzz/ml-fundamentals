#!/usr/bin/env python3
"""Entropy is a measure of disorder. In other words,
Entropy is an indicator of how messy your data is.

Let us imagine we have a set of N items.
These items fall into two categories, n have Label 1
and m=N-n have Label 2. As we have seen, to get
our data a bit more ordered, we want to group them by labels.
We introduce the ratio p=n/N and q=m/N=1-p

The entropy of out set is given by the following equation:

 E = -p lob2(p) - q log2(q)

The entropy is an absolute measure which provides a number between 0 and 1,
independently of the size of the set. At least if we consider the
binary scenario and log with base 2.
"""

import fileinput
import string
from math import log


def _identity(value): return value


def range_bytes(): return range(256)


def range_binary(): return range(2)


def range_printable(): return (ord(c) for c in string.printable)


def H(data, iterator=range_bytes, convert=_identity, base=2):
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(convert(x))) / len(data)
        if p_x > 0:
            # entropy += - p_x * math.log1p(p_x)
            # entropy += - p_x * math.log(p_x)
            # entropy += - p_x * math.log(p_x, 2)
            # entropy += - p_x * log(p_x, _resolve_base(data))
            entropy += - p_x * log(p_x, base)
    return entropy


# Shannon’s entropy
def shannon_entropy(data):
    return H(data, range_binary)


def _resolve_base(data):
    base = len(set(data))
    if base < 2:
        base = 2
    return base


def main():
    for row in fileinput.input():
        string = row.rstrip('\n')
        print("%s: %f" % (string, H(string, range_printable)))


# for str in ['gargleblaster', 'tripleee', 'magnus', 'lkjasdlk',
#             'aaaaaaaa', 'sadfasdfasdf', '7&wS/p(', 'aabb']:
#     print("%s: %f" % (str, H(str, range_printable, chr)))


def column(matrix, i):
    return [row[i] for row in matrix]


dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]

# str_data = ''.join(str(e) for e in column(dataset, 2))
# print(str_data)
# print("%s: %f" % ("dataset", H(column(dataset, 2), range_binary)))

dataset1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
dataset2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

print("%s: %f" % ("dataset 1", H(dataset1, range_binary)))
print("%s: %f" % ("dataset 2", H(dataset2, range_binary)))

print("%s: %f" % ("dataset 1", shannon_entropy(dataset1)))
print("%s: %f" % ("dataset 2", shannon_entropy(dataset2)))
