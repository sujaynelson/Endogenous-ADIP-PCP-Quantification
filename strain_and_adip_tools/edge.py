import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from strain_and_adip_tools import junction
from strain_and_adip_tools.circle_fit import fit


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def collinear(points, epsilon=0.01):
    """ determine if three points are collinear

    :param epsilon:
    :param points: list of 3 point tuples that might be collinear
    :type points: list of 3 2-member tuples
    :return: boolean to tell if the points are collinear or not
    :rtype: bool
    """
    if not isinstance(points, list):
        raise TypeError("parameter must be of type 'list'")

    if len(points) != 3:
        raise ValueError("list of points must have only 3 points")

    # converts points to a np array that looks like:
    # | x1 x2 x3 |
    # | y1 y2 y3 |
    # | 1  1  1  |
    matrix = np.array(
        [[points[0][0], points[1][0], points[2][0]], [points[0][1], points[1][1], points[2][1]], [1, 1, 1]])

    # finds area of the triangle formed by the three points by taking the determinant of the matrix above
    area = 0.5 * abs(np.linalg.det(matrix))

    if area < epsilon:
        return True
    else:
        return False


class Edge:
    id_iter = itertools.count()

    def __init__(self, start_node, end_node, intermediate_points, cells):
        self._start_node = start_node
        self._end_node = end_node
        self._radius = 0
        self._center = (0, 0)
        self._intermediate_points = intermediate_points
        self._mesh_segments = []
        self._mesh_points = []
        self._junctions = {start_node, end_node}
        self._cell_label_set = cells
        self._label = next(Edge.id_iter)
        self._corresponding_tension_vector = None
        self.tension_label = 0
        self.tension_magnitude = 0

    def _get_split_point(self, a, b, dist):

        """ Returns the point that is <<dist>> length along the line a b.

        a and b should each be an (x, y) tuple.
        dist should be an integer or float, not longer than the line a b.

        """

        dx = b[0] - a[0]
        dy = b[1] - a[1]

        try:
            m = dy / dx
        except ZeroDivisionError:
            if b[1] > a[1]:
                return a[0], a[1] + dist
            elif a[1] > b[1]:
                return b[0], b[1] + dist
            else:
                return a
        c = a[1] - (m * a[0])

        x = a[0] + (dist ** 2 / (1 + m ** 2)) ** 0.5
        y = m * x + c
        # formula has two solutions, so check the value to be returned is
        # on the line a b.
        if not (a[0] <= x <= b[0]) and (a[1] <= y <= b[1]):
            x = a[0] - (dist ** 2 / (1 + m ** 2)) ** 0.5
            y = m * x + c

        return x, y

    def split_line_single(self, line, length):

        """ Returns two ogr line geometries, one which is the first length
        <<length>> of <<line>>, and one one which is the remainder.

        line should be a ogr LineString Geometry.
        length should be an integer or float.

        """

        line_points = line
        sub_line = []
        while length > 0:
            d = distance(line_points[0], line_points[1])
            if d > length:
                split_point = self._get_split_point(line_points[0], line_points[1], length)
                sub_line.append(line_points[0])
                sub_line.append(split_point)
                line_points[0] = split_point
                break

            if d == length:
                sub_line.append(line_points[0])
                sub_line.append(line_points[1])
                line_points.remove(line_points[0])
                break

            if d < length:
                sub_line.append(line_points[0])
                line_points.remove(line_points[0])
                length -= d

        remainder = []
        for point in line_points:
            remainder.append(point)

        return sub_line, remainder

    def split_line_multiple(self, length=None, n_pieces=None):

        """ Splits a ogr wkbLineString into multiple sub-strings, either of
        a specified <<length>> or a specified <<n_pieces>>.

        line should be an ogr LineString Geometry
        Length should be a float or int.
        n_pieces should be an int.
        Either length or n_pieces should be specified.

        Returns a list of ogr wkbLineString Geometries.

        """
        line = self._intermediate_points.copy()
        if not n_pieces:
            n_pieces = int(math.ceil(self.length / length))
        if not length:
            length = self.length / float(n_pieces)

        line_segments = []
        remainder = line

        for i in range(n_pieces - 1):
            segment, remainder = self.split_line_single(remainder, length)
            line_segments.append(segment)
        else:
            line_segments.append(remainder)

        self._mesh_segments = line_segments

    def calculate_edge_points(self):
        for segment in self._mesh_segments:
            self._mesh_points.append(segment[0])
        self._mesh_points.append(self._mesh_segments[-1][-1])

    def outside(self, background):
        return background in self._cell_label_set

    @property
    def start_node(self):
        return self._start_node

    @start_node.setter
    def start_node(self, node):
        if isinstance(node, junction.Junction):
            self._start_node = node
        else:
            raise TypeError('node should be of type Junction. Instead, node was of type {}'.format(type(node)))

    @property
    def end_node(self):
        return self._end_node

    @end_node.setter
    def end_node(self, node):
        if isinstance(node, junction.Junction):
            self._end_node = node
        else:
            raise TypeError('node should be of type Junction. Instead, node was of type {}'.format(type(node)))

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        if isinstance(r, (int, float, complex)) and not isinstance(r, bool):
            self._radius = r
        else:
            raise TypeError('radius must be of numeric type. Instead, r was of type {}'.format(type(r)))

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, c):
        if len(c) == 2:
            self._center = c
        else:
            raise ValueError('center should not exceed length of 2. The length of center coordinates was: {}'.format(
                len(c)))

    @property
    def xc(self):
        return self._center[0]

    @property
    def yc(self):
        return self._center[1]

    @property
    def corresponding_tension_vector(self):
        return self._corresponding_tension_vector

    @corresponding_tension_vector.setter
    def corresponding_tension_vector(self, tension_vector):
        if isinstance(tension_vector, tension_vector.TensionVector):
            self._corresponding_tension_vector = tension_vector
        else:
            raise TypeError('corresponding_edge should be of type TensionVector. Instead, it was of type {}'.format(
                type(tension_vector)))

    @property
    def length(self):
        length = 0
        for index, point in enumerate(self._intermediate_points):
            if index < len(self._intermediate_points) - 1:
                length += distance(point, self._intermediate_points[index + 1])
        return length

    @property
    def location(self):
        return self._intermediate_points[int(len(self._intermediate_points) / 2)]

    def plot(self, label=False):
        x, y = list(zip(*self._intermediate_points))
        x = list(x)
        y = list(y)
        plt.plot(x, y, color='deepskyblue', linestyle='-', linewidth=0.5)
        if label:
            plt.text(self.location[0], self.location[1], str(self._label), color='white', fontsize=3,
                     horizontalalignment='center', verticalalignment='center')

    def circle_fit(self):
        x, y = list(zip(*self._mesh_points))
        x = np.asarray(x)
        y = np.asarray(y)
        xc, yc, radius = fit(x, y, self.start_node.coordinates, self.end_node.coordinates)
        self.center = (xc, yc)
        self.radius = radius

    @property
    def linear(self):
        for index, point in enumerate(self._mesh_points):
            if index < len(self._mesh_points) - 3:
                l = [point, self._mesh_points[index + 1], self._mesh_points[index + 2]]
                if not collinear(l):
                    return False
        return True

    def angular_position(self, coordinates):
        """ given a (x,y) coordinate, the angular position in radians from 0 to 2*pi around the fit circle is returned

        :param coordinates:
        :return:
        """
        x, y = coordinates
        angular_position = math.atan2(y - self.yc, x - self.xc)  # y,x
        # convert to radians between 0 and 2*pi
        if angular_position > 0:
            angular_position = angular_position
        else:
            angular_position = 2 * math.pi + angular_position
        return angular_position

    @property
    def cw_around_circle(self):
        """true if the edge (start_node to end_node) goes clockwise around the fit circle, false if ccw

        :return:
        """

        start_angular_position = self.angular_position(self.start_node.coordinates)
        end_angular_position = self.angular_position(self.end_node.coordinates)
        return start_angular_position > end_angular_position

    def start_tangent_angle(self, method='chord'):
        if method == 'chord':
            angle = math.atan2(self.end_node.y - self.start_node.y, self.end_node.x - self.start_node.x)
            # convert to range of 0 to 2pi
            if angle < 0:
                angle += 2 * math.pi
            return angle

        if method == 'circular':
            # circular edges
            slope_of_tangent_line = (-(self.start_node.x - self.xc)) / (self.start_node.y - self.yc)
            start_angular_position = self.angular_position(self.start_node.coordinates)
            angle = math.atan(slope_of_tangent_line)
            if self.cw_around_circle:
                if math.pi < start_angular_position < 2 * math.pi:
                    # third and fourth quadrants
                    angle += math.pi
            else:
                if math.pi > start_angular_position > 0:
                    # first and second quadrants
                    angle += math.pi

        # linear edges
        if self.linear:
            if self.end_node.x > self.start_node.x:
                angle = math.atan2(-self.start_node.y + self.end_node.y, -self.start_node.x + self.end_node.x)
            elif self.end_node.x == self.start_node.x:
                if self.end_node.y - self.start_node.y > 0:
                    angle = math.pi / 2
                else:
                    angle = 3 * math.pi / 2
            elif self.end_node.x < self.start_node.x:
                if self.end_node.y > self.start_node.y:
                    angle = math.pi / 2 + math.atan(
                        (self.start_node.x - self.end_node.x) / (-self.start_node.y + self.end_node.y))
                else:
                    angle = math.pi + math.atan(
                        (self.start_node.y - self.end_node.y) / (self.start_node.x - self.end_node.x))

        # convert to range of 0 to 2pi
        if angle < 0:
            angle += 2 * math.pi
        return angle

    def end_tangent_angle(self, method='chord'):
        if method == 'chord':
            angle = math.atan2(-self.end_node.y + self.start_node.y, -self.end_node.x + self.start_node.x)
            # convert to range of 0 to 2pi
            if angle < 0:
                angle += 2 * math.pi
            return angle

        if method == 'circular':
            slope_of_tangent_line = (-(self.end_node.x - self.xc)) / (self.end_node.y - self.yc)
            end_angular_position = self.angular_position(self.end_node.coordinates)
            angle = math.atan(slope_of_tangent_line)
            if self.cw_around_circle:
                if math.pi < end_angular_position < 2 * math.pi:
                    # third and fourth quadrants
                    angle += math.pi
            else:
                if math.pi > end_angular_position > 0:
                    # first and second quadrants
                    angle += math.pi

            # convert to range of 0 to 2pi
            if angle < 0:
                angle += 2 * math.pi
            angle -= math.pi

        # linear edges
        if self.linear:
            if self.start_node.x > self.end_node.x:
                angle = math.atan2(-self.end_node.y + self.start_node.y, -self.end_node.x + self.start_node.x)
            elif self.start_node.x == self.end_node.x:
                if self.start_node.y - self.end_node.y > 0:
                    angle = math.pi / 2
                else:
                    angle = 3 * math.pi / 2
            elif self.start_node.x < self.end_node.x:
                if self.start_node.y > self.end_node.y:
                    angle = math.pi / 2 + math.atan(
                        (self.end_node.x - self.start_node.x) / (-self.end_node.y + self.start_node.y))
                else:
                    angle = math.pi + math.atan(
                        (self.end_node.y - self.start_node.y) / (self.end_node.x - self.start_node.x))

            # convert to range of 0 to 2pi
            if angle < 0:
                angle += 2 * math.pi

        return angle

    def map_unit_vectors_to_junctions(self):
        angle = self.start_tangent_angle()

        self.start_node.x_unit_vectors_dict[self._label] = math.cos(angle)
        self.start_node.y_unit_vectors_dict[self._label] = math.sin(angle)

        angle = self.end_tangent_angle()

        self.end_node.x_unit_vectors_dict[self._label] = math.cos(angle)
        self.end_node.y_unit_vectors_dict[self._label] = math.sin(angle)

    def plot_tangent(self, c='y', ):
        angle = self.start_tangent_angle()

        plt.plot([self.start_node.x, self.start_node.x + 10 * math.cos(angle)], [self.start_node.y,
                                                                                 self.start_node.y + 10 * math.sin(
                                                                                     angle)], c=c, lw=0.75)

        angle = self.end_tangent_angle()
        plt.plot([self.end_node.x, self.end_node.x + 10 * math.cos(angle)], [self.end_node.y,
                                                                             self.end_node.y + 10 * math.sin(
                                                                                 angle)], c=c, lw=0.75)
        plt.text(self.location[0], self.location[1], str(round(self.tension_magnitude, 2)), color='red',
                 fontsize=3,
                 horizontalalignment='center', verticalalignment='center')
    def plot_circle(self):
        xc, yc = self._center
        start_point = self.start_node.coordinates
        mid_point = self._mesh_points[1]
        end_point = self.end_node.coordinates
        start_theta = math.atan2(start_point[1] - yc, start_point[0] - xc)
        end_theta = math.atan2(end_point[1] - yc, end_point[0] - xc)
        mid_theta = math.atan2(mid_point[1] - yc, mid_point[0] - xc)
        if start_theta <= mid_theta <= end_theta:
            theta_fit = np.linspace(start_theta, end_theta, 100)
        elif start_theta >= mid_theta >= end_theta:
            theta_fit = np.linspace(end_theta, start_theta, 100)
        else:
            if start_theta < 0:
                start_theta = 2 * math.pi + start_theta
            if end_theta < 0:
                end_theta = 2 * math.pi + end_theta
            if mid_theta < 0:
                mid_theta = 2 * math.pi + mid_theta
            theta_fit = np.linspace(start_theta, end_theta, 180)

        # stores all x and y coordinates along the fitted arc
        x_fit = xc + self.radius * np.cos(theta_fit)
        y_fit = yc + self.radius * np.sin(theta_fit)

        # plot least squares circular arc
        plt.plot(x_fit, y_fit, 'r-', lw=1)

    def __eq__(self, other):
        return self._junctions == {other.start_node, other.end_node}

    def __str__(self):
        return str(self._start_node) + ' to ' + str(self._end_node)

    def __hash__(self):
        return hash(str(self))
