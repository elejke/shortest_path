from embedding import *
from geometry import *

# task constants
PIN_NUMBER = 40


# quality minimum estimations:
L_min = 2
V_min = 40
S_min = 580

# optimization formula :  K = L_min/L + 2 * V_min/V + 3 * S_min/S


def objective(connections, subsequences, subsequences_by_layers, chip_1, chip_2):
    S = 0.0
    V = 0.0
    L = 0.0

    for (layer, seq) in enumerate(subsequences_by_layers):
        internal, external, jump_lines = get_lines(connections, subsequences[seq[0]], subsequences[seq[1]], chip_1, chip_2, layer+1)
        internal, external = optimize_embedding(internal, external)

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

    K = L_min / L + 2 * V_min / V + 3 * S_min / S

    return K, L, V, S