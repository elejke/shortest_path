import numpy as np
# import pandas as pd
import matplotlib.pylab as plt
from sys import maxint

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import LinearRing
from descartes.patch import PolygonPatch


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
    >>>_line_to_linestring([(0, 0), (0, 1), (1, 1)]).length
    2.0
    """
    return LineString(raw_line)


def sum_length(raw_lines):
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
    print 'POLY ' + str(points_count) + ' ' + str(layer)
    for point in points:
        print str(point[0]) + ', ' + str(point[1])


def _plot_coords(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, color='#999999', zorder=1)


def plot_all_lines(raw_lines, distance=0.05):
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
    
    # GOVNO
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            dist = LinearRing(np.array(polygons[i].exterior.xy).T).distance(
                LinearRing(np.array(polygons[j].exterior.xy).T))
            if min_dist > dist:
                min_dist = dist
                
    return min_dist
