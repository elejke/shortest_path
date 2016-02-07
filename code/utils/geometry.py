import numpy as np
# import pandas as pd
import matplotlib.pylab as plt

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import LinearRing
from descartes.patch import PolygonPatch

from sys import maxint
import sys


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def line_to_poly(raw_line, distance=0.05):
    """
    Line in format [(), ..., ()] to Polygon with 2*distance width
    Args:
        raw_line (list): Connected dots
        distance (Optional[float]): width of polygon = 2*distance
    """
    line = LineString(raw_line)
    return Polygon(line.buffer(distance, cap_style=2, join_style=2))


def _line_to_linestring(raw_line):
    """
    Line in format [(), ..., ()] to LineString
    Args:
        raw_line (list): Connected dots
    Example:
    >>>_line_to_linestring([(0, 0), (0, 1), (1, 1), (1, 1)]).length
    2.0
    """
    return LineString(raw_line)


def sum_length(raw_lines):
    """
    Find summarize length of all lines:

    Args:
        raw_lines (list): list of lines if format [(x_1, y_1), ..., (x_n, y_n)]
    """
    lines_lengths = [_line_to_linestring(raw_line).length for raw_line in raw_lines]
    return np.sum(lines_lengths)


def print_poly(poly, layer):
    """
    Prints polygon in test-like style
    Args:
        poly (Polygon): Polygon to print
        layer (int): The number of the layer
    """
    points = np.array(poly.exterior.coords.xy).T[:-1]
    points_count = len(points)
    print 'POLY ' + str(points_count) + ', ' + str(layer)
    for point in points:
        print str(point[0]).replace(".", ",") + '; ' + str(point[1]).replace(".", ",")


def print_jumps(jumps):
    """
    Print jumps in test-like style

    Args:
        jumps (list): list of coordinates in format [(x_1, y_1), ..., (x_n, y_n)]
    """
    for jump in jumps:
        print 'JUMP ' + str(jump[0]).replace(".", ",") + '; ' + str(jump[1]).replace(".", ",")


def submit(int_lines_list, ext_lines_list, jump_lines, jumps):
    temp = sys.stdout
    sys.stdout = open("ans", "w")

    for layer, int_lines in enumerate(int_lines_list):
        for line in int_lines:
            print_poly(line_to_poly(line), layer+1)
    for layer, ext_lines in enumerate(ext_lines_list):
        for line in ext_lines:
            print_poly(line_to_poly(line), layer+1)
    for line in jump_lines:
        print_poly(line_to_poly(line), 1)

    print_jumps(jumps)

    sys.stdout.close()
    sys.stdout = temp


def _plot_coords(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, color='#999999', zorder=1)


def plot_all_lines(raw_lines, distance=0.05):
    """
    Plot lines as polygons with width = 2 * distance

    Args:
        raw_lines (list): list of lines if format [(x_1, y_1), ..., (x_n, y_n)]
        distance (float): half of minimal distance between every dots of polygon
    """
    fig = plt.figure(1, figsize=[10, 10], dpi=90)
    
    ax = fig.add_subplot(1, 1, 1)
    
    for raw_line in raw_lines:
        line = _line_to_linestring(raw_line)
        polygon = line.buffer(distance, cap_style=2, join_style=2)
        
        _plot_coords(ax, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor='blue', edgecolor='blue', alpha=0.5)
        ax.add_patch(patch)


def min_distance(raw_lines):
    min_dist = maxint
    polygons = [line_to_poly(raw_line) for raw_line in raw_lines]

    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            dist = LinearRing(np.array(polygons[i].exterior.xy).T).distance(
                LinearRing(np.array(polygons[j].exterior.xy).T))
            if min_dist > dist:
                min_dist = dist

    return min_dist


def min_distance_to_pins(raw_lines, pins):
    min_dist = maxint
    # polygons = [line_to_poly(raw_line) for raw_line in raw_lines]

    for line in raw_lines:
        for pin in pins:
            dist = LinearRing(np.array(line_to_poly(line).exterior.xy).T).distance(
                LinearRing(np.array(Point(pin).buffer(0.1).exterior.xy).T))

            if min_dist > dist:
                if not (is_close(pin[0], line[0][0], abs_tol=0.00000000001) and
                            is_close(pin[1], line[0][1])) and not (
                            is_close(pin[0], line[-1][0], abs_tol=0.00000000001) and\
                                is_close(pin[1], line[-1][1])):
                    min_dist = dist
                    # print 'min_dist : ', min_dist

    return min_dist
