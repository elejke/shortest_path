import numpy as np
import pandas as pd

def data_prepare(coordinates_file='../data/coordinates.csv', connections_file='../data/connect.csv'):
    """lol"""
    coordinates = pd.read_csv(coordinates_file, ';', header=None, decimal=',')
    connections = pd.read_csv(connections_file, ';', header=None) - 1
    chip_1 = coordinates[:40].drop([0, 1], axis=1)
    chip_2 = coordinates[40:].drop([0, 1], axis=1)
    return chip_1, chip_2, connections #, coordinates

def min_distance(chip_1, chip_2, connections):
    """
    Minimal distance estimation
    """
    min_distance = 0.0
    for i in connections.values:
        min_distance += np.sqrt(np.sum((chip_1.values[i[0]-1]-chip_2.values[i[1]-1])**2))
    return min_distance

def lis(X):
    """
    Find and return longest increasing subsequence of S.
    If multiple increasing subsequences exist, the one that ends
    with the smallest value is preferred, and if multiple
    occurrences of that value can end the sequence, then the
    earliest occurrence is preferred.
    """
    n = len(X)
    X = [None] + X  # Pad sequence so that it starts at X[1]
    M = [None]*(n+1)  # Allocate arrays for M and P
    P = [None]*(n+1)
    L = 0
    for i in range(1,n+1):
        if L == 0 or X[M[1]] >= X[i]:
            # there is no j s.t. X[M[j]] < X[i]]
            j = 0
        else:
            # binary search for the largest j s.t. X[M[j]] < X[i]]
            lo = 1      # largest value known to be <= j
            hi = L+1    # smallest value known to be > j
            while lo < hi - 1:
                mid = (lo + hi)//2
                if X[M[mid]] < X[i]:
                    lo = mid
                else:
                    hi = mid
            j = lo

        P[i] = M[j]
        if j == L or X[i] < X[M[j+1]]:
            M[j+1] = i
            L = max(L,j+1)

    # Backtrack to find the optimal sequence in reverse order
    values = []
    indexes = []
    pos = M[L]
    while L > 0:
        values.append(X[pos])
        indexes.append(pos-1)
        pos = P[pos]
        L -= 1

    indexes.reverse()
    values.reverse()
    return indexes, values
    
def layers(chip_1, chip_2, connections):
    """TODO"""
    new_points = []
    for num, connect in enumerate(connections.values):
        if chip_2.values[connect[1]][0] < 14.5:
            new_points.append([chip_1.values[connect[0]][0], chip_2.values[connect[1]][1], num])
        if chip_2.values[connect[1]][0] > 14.5:
            new_points.append([chip_1.values[connect[0]][0], chip_2.values[connect[1]][1]+0.3, num])

    residuals = new_points
    residuals.sort()
    residuals.reverse()

    subsequences = []
    while len(residuals) > 0:
        sequence = list(zip(*residuals)[1])
        longest_subseq = lis(sequence)
        connection_idxs = np.array(zip(*np.array(residuals)[longest_subseq[0]].tolist())[2], dtype=np.int).tolist()
        subsequences.append(connection_idxs)

        idxs = list(set(range(len(residuals))) - set(longest_subseq[0]))
        idxs.sort()
        residuals = np.array(residuals)[idxs].tolist()
    
    return new_points, subsequences
