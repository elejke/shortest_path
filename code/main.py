import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from utils.embedding import *
from utils.geometry import *
from utils.constants import *
from shapely.geometry import Point, LineString, Polygon, LinearRing

chip_1, chip_2, connections = data_prepare()

new_points, subsequences = layers(chip_1, chip_2, connections)

subsequences_by_layers_1 = [[0, 1], [4, 7], [3, 6], [2, 5]]
subsequences_by_layers_2 = [[0, 7], [1, 6], [2, 5], [3, None], [4, None]]

K, L, V, S, mind, int_lines_list, ext_lines_list = \
        objective(connections, subsequences, subsequences_by_layers_1, chip_1, chip_2)

jump_coordinates, jump_lines = get_jumps(connections, subsequences[subsequences_by_layers_1[0][0]],
                                         subsequences[subsequences_by_layers_1[0][1]], chip_1, chip_2, 1)

submit("submission", int_lines_list, ext_lines_list, jump_lines, jump_coordinates)

print "K = ", K
print "L = ", L
print "V = ", V
print "S = ", S
print "mind = ", mind
