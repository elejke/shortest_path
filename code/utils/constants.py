from embedding import *
from geometry import *
from sys import maxint

# task constants
PIN_NUMBER = 40


# quality minimum estimations:
L_min = 3
V_min = 32
S_min = 600

# optimization formula :  K = L_min/L + 2 * V_min/V + 3 * S_min/S


def objective(connections, subsequences, subsequences_by_layers, chip_1, chip_2):
    S = 0.0
    V = 0.0
    L = 0.0
    min_d = maxint
    int_lines_list = []
    ext_lines_list = []

    subsequences.append([])
    for (layer, seq) in enumerate(subsequences_by_layers):
        was_none = False
        if seq[1] is None:
            seq[1] = len(subsequences)-1
            was_none = True
        internal, external, jump_lines = get_lines(connections,
                                                   subsequences[seq[0]],
                                                   subsequences[seq[1]],
                                                   chip_1,
                                                   chip_2,
                                                   layer+1)
        internal, external = optimize_embedding(internal, external, jump_lines, chip_1, chip_2)
        int_lines_list.append(internal)
        ext_lines_list.append(external)

        if was_none:
            seq[1] = None

        all_lines = []
        for line in internal:
            all_lines.append(line)
        for line in external:
            all_lines.append(line)
        for line in jump_lines:
            all_lines.append(line)

        L += 1
        V += len(jump_lines)
        S += sum_length(all_lines)

        if layer == 0:
            min_d = min(min_d, min_distance(all_lines),
                        min_distance_to_pins(all_lines, chip_1.values),
                        min_distance_to_pins(all_lines, chip_2.values))
        else:
            min_d = min(min_d, min_distance(all_lines))

        print min_distance(all_lines), min_d

    subsequences.pop()
    K = L_min / L + 2 * V_min / V + 3 * S_min / S

    return K, L, V, S, min_d, int_lines_list, ext_lines_list
