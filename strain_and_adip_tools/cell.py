import math
from itertools import combinations

import matplotlib.pyplot as plt

from strain_and_adip_tools.edge import Edge
from strain_and_adip_tools.path_finder import breadth_first_search


class Cell:

    def __init__(self, pixel_value):
        """ constructor for a Cell object

        :param pixel_value: value of all of pixels that make up this Cell in the array
        :type pixel_value: float
        """
        # identify each cell based on its pixel value
        self._label = pixel_value

        # set of tuples of points in cell boundary
        # each element looks like ((x, y), neighboring_cell_label)
        self._edge_point_set = set()

        self._cell_boundary_segments = []
        self.junctions = set()

    @property
    def neighboring_cell_labels(self):
        neighboring_cell_labels = set()
        for j in self.junctions:
            neighboring_cell_labels.update(j.cell_labels)
        return neighboring_cell_labels

    def add_edge_point(self, edge_point, neighboring_cell_label):
        self._edge_point_set.add((edge_point, neighboring_cell_label))

    def generate_maze(self, neighboring_cell_label):
        xmin = 212
        ymin = 212
        xmax = 0
        ymax = 0
        for (x, y), neighboring_cell in self._edge_point_set:
            if x > xmax:
                xmax = x
            if x < xmin:
                xmin = x
            if y > ymax:
                ymax = y
            if y < ymin:
                ymin = y
        rows, cols = (int(ymax - ymin + 1), int(xmax - xmin + 1))
        arr = [[0 for i in range(cols)] for j in range(rows)]
        for (x, y), neighboring_cell in self._edge_point_set:
            if neighboring_cell == neighboring_cell_label:
                row = int(y - ymin)
                col = int(x - xmin)
                arr[row][col] = 1
        for junction in self.junctions:
            if neighboring_cell_label in junction.cell_labels:
                x, y = junction.coordinates
                row = int(y - ymin)
                col = int(x - xmin)
                if row < 0 or col < 0:
                    continue
                if row >= rows or col >= cols:
                    continue
                
                arr[row][col] = 1
        return arr, xmin, ymin

    def make_edges(self, master_set):

        for start, end in combinations(self.junctions, 2):
            already_created_edge = False
            for edge in master_set:
                if edge.start_node == start and edge.end_node == end:
                    already_created_edge = True
                if edge.start_node == end and edge.end_node == start:
                    already_created_edge = True
            if not already_created_edge:
                start_cells = start.cell_labels
                end_cells = end.cell_labels
                shared_cell_labels = start_cells.intersection(end_cells)
                if len(shared_cell_labels) == 2:
                    (neighboring_cell_label,) = shared_cell_labels - {self.label}
                    maze, xmin, ymin = self.generate_maze(neighboring_cell_label)
                    x1, y1 = start.coordinates
                    point1 = (int(y1 - ymin), int(x1 - xmin))

                    x2, y2 = end.coordinates
                    point2 = (int(y2 - ymin), int(x2 - xmin))

                    path = breadth_first_search(maze, point1, point2)

                    path_new = []
                    if path is None:
                        continue
                    for point in path:
                        y, x = tuple(map(lambda i, j: i + j, point, (ymin, xmin)))
                        path_new.append((x, y))

                    self._cell_boundary_segments.append(path_new)
                    e = Edge(start, end, path_new, {self.label, neighboring_cell_label})
                    master_set.add(e)

    @property
    def number_of_edge_points(self):
        """ returns the number of edge points in edge_point_list

        :return: number of edge points
        """

        return len(self._edge_point_set)

    @property
    def label(self):
        """ the label of a Cell is it's unique pixel value. It is assigned when the Cell object is created.

        :return:
        """
        return self._label

    def approximate_cell_center(self):
        """ approximates the coordinates of the center of the cell by averaging the coordinates of points on the
        perimeter (edge) of the cell

        :return approximate center of the cell
        :rtype: tuple
        """

        xsum = 0
        ysum = 0
        for point, neighbor in self._edge_point_set:
            xsum += point[0]
            ysum += point[1]
        xc = xsum / len(self._edge_point_set)
        yc = ysum / len(self._edge_point_set)
        return xc, yc

    def plot(self):
        plt.text(self.approximate_cell_center()[0], self.approximate_cell_center()[1], str(self._label), color='green',
                 fontsize=3,
                 horizontalalignment='center', verticalalignment='center')

    def __str__(self):
        return str('Cell {}'.format(self._label))

    def __repr__(self):
        return repr('Cell {}'.format(self._label))

    def __eq__(self, other):
        return math.isclose(self.label, other.label)

    def __hash__(self):
        return hash(str(self))
