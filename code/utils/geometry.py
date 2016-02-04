import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from shapely.geometry import Polygon
from shapely.geometry import LineString
from descartes.patch import PolygonPatch

def line_to_poly(line, distance=0.05):
    """
    Line in format [(x_1, y_1), ..., (x_n, y_n)] to Polygon with 2*distance width
    """
    return Polygon(line.buffer(distance, cap_style=2, join_style=2))

def print_poly(poly, layer):
    """Prints polygon in test-like style"""
    points = np.array(poly.exterior.coords.xy).T[:-1]
    points_count = len(points)
    print 'POLY ' + str(points_count) + ' ' + str(layer)
    for point in points:
        print str(point[0]) + ', ' + str(point[1])
        
def plot_coords(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, color='#999999', zorder=1)
        
def plot_all_lines(raw_lines):
    fig = plt.figure(1, figsize=[10, 10], dpi=90)
    
    ax = fig.add_subplot(1, 1, 1)
    lines = []
    
    for raw_line in raw_lines:
        line = LineString(raw_line)
        lines.append(line)
        polygon = line.buffer(0.05, cap_style=2, join_style=2)
        
        plot_coords(ax, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor='blue', edgecolor='blue', alpha=0.5)
        ax.add_patch(patch)

def min_distance(raw_lines):
    min_dist = 100000
    lines = []
    
    for raw_line in raw_lines:
        line = LineString(raw_line)
        lines.append(line)
    
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            if min_dist > lines[i].distance(lines[j]):
                min_dist = lines[i].distance(lines[j])
                
    return min_dist