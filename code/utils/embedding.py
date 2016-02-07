import numpy as np
import pandas as pd

from geometry import *


def data_prepare(coordinates_file='../data/coordinates.csv', connections_file='../data/connect.csv'):
    """
    Load and separete data by chips

    Args:
        coordinates_file (file): file with coordinates of pins
        connections_file (file): file with required connections between pins
    """
    coordinates = pd.read_csv(coordinates_file, ';', header=None, decimal=',')
    connections = pd.read_csv(connections_file, ';', header=None) - 1
    chip_1 = coordinates[:40].drop([0, 1], axis=1)
    chip_2 = coordinates[40:].drop([0, 1], axis=1)
    return chip_1, chip_2, connections


def lis(X):
    # EXTERNAL CODE
    """
    Find and return longest increasing subsequence of S.
    If multiple increasing subsequences exist, the one that ends
    with the smallest value is preferred, and if multiple
    occurrences of that value can end the sequence, then the
    earliest occurrence is preferred.

    Args:
        X (list): sequence to find lis
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
        # find longest decreasing subsequence to determine the size of the minimal covering by increasing subsequences
        # test_sequence = map(lambda x: -x, sequence)
        # longes = lis(test_sequence)
        connection_idxs = np.array(zip(*np.array(residuals)[longest_subseq[0]].tolist())[2], dtype=np.int).tolist()
        subsequences.append(connection_idxs)

        idxs = list(set(range(len(residuals))) - set(longest_subseq[0]))
        idxs.sort()
        residuals = np.array(residuals)[idxs].tolist()
    
    return new_points, subsequences


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def embedding(connections, int_seq, ext_seq, chip_1, chip_2, layer):
    
    int_seq_lines = []
    ext_seq_lines = []
    jump_coordinates = []
    jump_lines = []

    down_shift = 0.101  # change in two places

    # internal lines formation:
    for connect in connections.values[int_seq]:
        if chip_2.values[connect[1]][0] < 14.5:
            x = [chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 14.0,
                 14.0,
                 14.0,
                 14.0 + 0.101,
                 14.0]
            y = [chip_1.values[connect[0]][1],
                 chip_1.values[connect[0]][1] - down_shift * int(layer > 1),
                 0.5,
                 chip_2.values[connect[1]][1],

                 #chip_2.values[connect[1]][1] - 0.15 + 0.01 * int(layer > 1),
                 #chip_2.values[connect[1]][1] - 0.15 + 0.01 * int(layer > 1),

                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1]]
        else:
            x = [chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 14.0,
                 chip_2.values[connect[1]][0] - 0.6,
                 chip_2.values[connect[1]][0] - 0.2,
                 chip_2.values[connect[1]][0] - 0.101,
                 chip_2.values[connect[1]][0]]
            y = [chip_1.values[connect[0]][1],
                 chip_1.values[connect[0]][1] - down_shift * int(layer > 1),
                 0.5,

                 # new: \/
                 chip_2.values[connect[1]][1] + 0.3,

                 # old chip_2.values[connect[1]][1] + 0.15 - 0.01 * int(layer > 1),
                 chip_2.values[connect[1]][1] + 0.3,

                 # old chip_2.values[connect[1]][1] + 0.15 - 0.01 * int(layer > 1),
                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1],

                 chip_2.values[connect[1]][1]]

        if layer > 1:
            x[0] = x[1]
            x[-1] = x[-2]
            y[0] = y[1]
            y[-1] = y[-2]

        int_seq_lines.append((x, y))
    
    x_turn = np.max(list(zip(*chip_1.values)[0])) + 0.11
    y_turn = np.min(list(zip(*chip_2.values)[1])) - 0.15
    
    # external lines formation:
    const = 0.29

    for num, connect in enumerate(connections.values[ext_seq]):
        if chip_2.values[connect[1]][0] < 14.5:
            x = [chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 x_turn,

                 # new \/
                 chip_2.values[connect[1]][0] + 0.8 + const*(num+1),

                 # old chip_2.values[connect[1]][0] + 0.8 + const*(num+1),
                 chip_2.values[connect[1]][0] + 0.8 + const*(num+1),

                 # old chip_2.values[connect[1]][0] + 0.8 + const*(num+1),
                 chip_2.values[connect[1]][0] + 0.6,

                 # old chip_2.values[connect[1]][0],
                 chip_2.values[connect[1]][0] + 0.2,

                 chip_2.values[connect[1]][0] + 0.101,

                 chip_2.values[connect[1]][0]]

            y = [chip_1.values[connect[0]][1],
                 chip_1.values[connect[0]][1] - down_shift * int(layer > 1),
                 0-const*(num+1),
                 0-const*(num+1),
                 y_turn,

                 #new \/
                 chip_2.values[connect[1]][1] - 0.3,

                 # old chip_2.values[connect[1]][1] - 0.15,
                 chip_2.values[connect[1]][1] - 0.3,

                 # old chip_2.values[connect[1]][1] - 0.15,
                 chip_2.values[connect[1]][1],

                 chip_2.values[connect[1]][1],

                 chip_2.values[connect[1]][1]]
        else:
            x = [chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 chip_1.values[connect[0]][0],
                 x_turn,
                 chip_2.values[connect[1]][0] + const * (num + 1),
                 chip_2.values[connect[1]][0] + const * (num + 1),
                 chip_2.values[connect[1]][0],
                 chip_2.values[connect[1]][0],
                 chip_2.values[connect[1]][0] - 0.101,
                 chip_2.values[connect[1]][0]]
            y = [chip_1.values[connect[0]][1],
                 chip_1.values[connect[0]][1] - down_shift * int(layer > 1),
                 0-const*(num+1),
                 0-const*(num+1),
                 y_turn,
                 # chip_2.values[connect[1]][1] + 0.15,
                 # chip_2.values[connect[1]][1] + 0.15,

                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1],
                 chip_2.values[connect[1]][1]]

        if layer > 1:
            x[0] = x[1]
            x[-1] = x[-2]
            y[0] = y[1]
            y[-1] = y[-2]
        
        ext_seq_lines.append((x, y))

    # jumps processing:

    if layer == 1:
        for connect in connections.values[list(set(range(40)) - set(ext_seq) - set(int_seq))]:
            jump_coordinates.append([chip_1.values[connect[0]][0], chip_1.values[connect[0]][1] - down_shift + 0.1*down_shift])
            jump_lines.append(
                ([chip_1.values[connect[0]][0], chip_1.values[connect[0]][0]],
                 [chip_1.values[connect[0]][1], chip_1.values[connect[0]][1] - down_shift])
            )
            if chip_2.values[connect[1]][0] < 14.5:
                jump_coordinates.append([chip_2.values[connect[1]][0] + 0.101,
                                         chip_2.values[connect[1]][1]])
                jump_lines.append(
                    ([chip_2.values[connect[1]][0], chip_2.values[connect[1]][0]],
                     [chip_2.values[connect[1]][1] + 0.101, chip_2.values[connect[1]][1]])
                )
            else:
                jump_coordinates.append([chip_2.values[connect[1]][0] - 0.101,
                                         chip_2.values[connect[1]][1]])
                jump_lines.append(
                    ([chip_2.values[connect[1]][0], chip_2.values[connect[1]][0]],
                     [chip_2.values[connect[1]][1] - 0.101, chip_2.values[connect[1]][1]])
                )

    return int_seq_lines, ext_seq_lines, jump_lines, jump_coordinates


def get_lines(connections, int_seq, ext_seq, chip_1, chip_2, layer):
    emb = embedding(connections, int_seq, ext_seq, chip_1, chip_2, layer)
    internal_lines = np.array([line.T for line in np.array(emb[0])])
    external_lines = np.array([line.T for line in np.array(emb[1])])
    jump_lines = np.array([line.T for line in np.array(emb[2])])
    
    return internal_lines, external_lines, jump_lines


def break_external_by_separator(internal_lines, external_lines, separator, chip_1, chip_2):
    x_turn = np.min(list(zip(*chip_1.values)[0])) - 0.21
    y_turn = np.max(list(zip(*chip_2.values)[1])) + 0.21

    shape_upper = (separator, 12, 2)
    shape_lower = (len(external_lines)-separator, 12, 2)
    upper_external = np.zeros(shape_upper)
    lower_external = np.zeros(shape_lower)

    const = 0.31
    down_shift = 0.101  # change in two places
    for num in range(separator):

        connect_begin = external_lines[num][:2]
        connect_end = external_lines[num][-4:]

        connect_inter = [[connect_begin[-1][0], 0 - 0.2*(num+1) - (down_shift-0.05)],
                         [x_turn - const*(num+1), 0 - 0.2*(num+1) - (down_shift-0.05)],
                         [x_turn - const*(num+1), 0.1],
                         [13.9, y_turn + const*(num+1)],
                         [14.8 + 0.25*(num+1), y_turn + const*(num+1)],
                         [14.8 + 0.25*(num+1), connect_end[0][1]]]

        upper_external[num] = np.append(np.append(connect_begin, connect_inter, axis=0), connect_end, axis=0)

    for num in range(len(external_lines)-separator):
        lower_external[num] = np.append(external_lines[num+separator],
                                        np.array([external_lines[num+separator][-1],
                                                  external_lines[num+separator][-1]]),
                                        axis=0)

    return lower_external, upper_external


def optimize_external_embedding(internal_lines, external_lines, chip_1, chip_2):
    internal_lines = np.array(sorted(internal_lines, key=lambda x: x[0][0]))
    external_lines = np.array(sorted(external_lines, key=lambda x: x[0][0]))

    lengths = np.zeros(len(external_lines)+1)

    for separator in range(len(external_lines)+1):
        lower_external, upper_external = break_external_by_separator(internal_lines,
                                                                     external_lines,
                                                                     separator,
                                                                     chip_1,
                                                                     chip_2)
        lengths[separator] = sum_length(lower_external) + sum_length(upper_external)

    optimal_separator = np.argmin(lengths)
    lower_external, upper_external = break_external_by_separator(internal_lines,
                                                                 external_lines,
                                                                 optimal_separator,
                                                                 chip_1,
                                                                 chip_2)
    external_lines = np.append(lower_external, upper_external, axis=0)

    return internal_lines, external_lines


def optimize_embedding(internal_lines, external_lines, jump_lines, chip_1, chip_2):

    internal_lines = np.array(sorted(internal_lines, key=lambda x: -x[0][0]))
    external_lines = np.array(sorted(external_lines, key=lambda x: -x[0][0]))

    for i in range(len(internal_lines)):

        # shifting line according to previous internal line
        current_distance_to_internal = min_distance(internal_lines[i - 1:i + 1])
        while i > 0 \
                and not isclose(current_distance_to_internal, 0.1, abs_tol=0.00000000001) \
                and current_distance_to_internal < 0.1:

            initial_internal = internal_lines[i][3][0]
            internal_lines[i][3][0] -= 0.001
            distance_1 = min_distance(internal_lines[i - 1:i + 1])
            internal_lines[i][3][0] = initial_internal

            initial_internal = internal_lines[i][2][1]
            internal_lines[i][2][1] += 0.001
            distance_2 = min_distance(internal_lines[i - 1:i + 1])
            internal_lines[i][2][1] = initial_internal

            if distance_1 > distance_2:
                internal_lines[i][3][0] -= 0.001
                current_distance_to_internal = distance_1
            else:
                internal_lines[i][2][1] += 0.001
                current_distance_to_internal = distance_2

        # shifting line according to external lines
        for j in range(len(external_lines)):

            current_distance_to_external = min_distance([external_lines[j], internal_lines[i]])
            while not isclose(current_distance_to_external, 0.1, abs_tol=0.00000000001)\
                    and current_distance_to_external < 0.1:

                initial_internal = internal_lines[i][3][0]
                internal_lines[i][3][0] -= 0.001
                distance_1 = min_distance([external_lines[j], internal_lines[i]])
                internal_lines[i][3][0] = initial_internal

                initial_internal = internal_lines[i][2][1]
                internal_lines[i][2][1] += 0.001
                distance_2 = min_distance([external_lines[j], internal_lines[i]])
                internal_lines[i][2][1] = initial_internal

                if distance_1 > distance_2:
                    internal_lines[i][3][0] -= 0.001
                    current_distance_to_external = distance_1
                else:
                    internal_lines[i][2][1] += 0.001
                    current_distance_to_external = distance_2

        # shifting line according to jump lines
        for j in range(len(jump_lines)):

            current_distance_to_jump = min_distance([jump_lines[j], internal_lines[i]])
            while not isclose(current_distance_to_jump, 0.1, abs_tol=0.00000000001)\
                    and current_distance_to_jump < 0.1:

                initial_internal = internal_lines[i][3][0]
                internal_lines[i][3][0] -= 0.001
                distance_1 = min_distance([jump_lines[j], internal_lines[i]])
                internal_lines[i][3][0] = initial_internal

                initial_internal = internal_lines[i][2][1]
                internal_lines[i][2][1] += 0.001
                distance_2 = min_distance([jump_lines[j], internal_lines[i]])
                internal_lines[i][2][1] = initial_internal

                if distance_1 > distance_2:
                    internal_lines[i][3][0] -= 0.001
                    current_distance_to_jump = distance_1
                else:
                    internal_lines[i][2][1] += 0.001
                    current_distance_to_jump = distance_2

    internal_lines, external_lines = optimize_external_embedding(internal_lines, external_lines, chip_1, chip_2)

    return internal_lines, external_lines


def get_jumps(connections, int_seq, ext_seq, chip_1, chip_2, layer):
    emb = embedding(connections, int_seq, ext_seq, chip_1, chip_2, layer)
    return emb[3], np.array([line.T for line in np.array(emb[2])])
