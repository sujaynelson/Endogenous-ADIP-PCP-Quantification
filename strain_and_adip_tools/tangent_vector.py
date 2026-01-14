""" A class to define tension vectors in CellFIT"""

import itertools
from math import pi, sin, cos, isclose

import strain_and_adip_tools.edge
import strain_and_adip_tools.junction


class TangentVector:
    id_iter = itertools.count()

    def __init__(self, initial_point, terminal_point, edge, junction):
        """ Constructor for a TangentVector object

        :param initial_point:
        :type initial_point: tuple
        :param terminal_point:
        :type terminal_point: tuple
        :param edge:
        :type edge: edge.Edge
        :param junction:
        :type junction: junction.Junction
        """
        self._magnitude = 0
        self._direction = 0
        self._corresponding_edge = edge
        self._corresponding_junction = junction
        self._label = next(TangentVector.id_iter)

    @property
    def magnitude(self):
        """ Magnitude of the tangent vector

        :return: magnitude
        """
        return self._magnitude

    @magnitude.setter
    def magnitude(self, value):
        """ Set the magnitude of the tension vector to a value

        :param value: value of the magnitude that is being set
        :type value: float
        :return: None
        """
        self._magnitude = value

    @property
    def direction(self, units='rad'):
        """ returns the direction of the tension vector in radians (or degrees) from the horizontal

        :param units: units for the direction, either 'rad' or 'deg'
        :type units: str
        :raises ValueError: is units is not 'rad' or 'deg
        :return: direction
        :rtype: float
        """

        if units == 'rad':
            return self._direction
        elif units == 'deg':
            return self._direction * 180 / pi
        else:
            raise ValueError("units should be 'deg' or 'rad'. Your units were {}.".format(units))

    @direction.setter
    def direction(self, value, units='rad'):
        """ sets the direction of the tension vector with an input of radians (default) or degrees

        :param value: direction of the vector in radians from the horizontal
        :type value: float
        :param units: units for the direction, either 'rad' or 'deg'
        :type units: str
        :raises ValueError: is units is not 'rad' or 'deg
        :return: None
        """
        if units == 'rad':
            self._direction = value
        elif units == 'deg':
            self._direction = value * pi / 180
        else:
            raise ValueError("units should be 'deg' or 'rad'. Your units were {}.".format(units))

    @property
    def corresponding_edge(self):
        """ The corresponding Edge for this tension vector

        :return: corresponding edge
        :rtype: edge.Edge
        """

        return self._corresponding_edge

    @corresponding_edge.setter
    def corresponding_edge(self, edge):
        """ assigns a corresponding Edge for this tension vector

        :param edge: the corresponding edge for this tension vector
        :return: None
        """

        if isinstance(edge, edge.Edge):
            self._corresponding_edge = edge
        else:
            raise TypeError('corresponding_edge should be of type Edge. Instead, it was of type {}'.format(type(
                edge)))

    @property
    def label(self):
        """ returns the label (id number) for this tension vector

        :return: label
        :rtype: int
        """

        return self._label

    @property
    def x_component(self):
        """ returns the x-component of the tension vector

        :return: x-component of the vector
        :rtype: float
        """

        return self._magnitude * cos(self._direction)

    @property
    def y_component(self):
        """ returns the y-component of the tension vector

        :return: y-component of the vector
        :rtype: float
        """

        return self._magnitude * sin(self._direction)

    def __eq__(self, other):
        return isclose(self._magnitude, other.magnitude) and isclose(self._direction, other.direction)

    def __str__(self):
        return str(self._magnitude) + ' @ ' + str(self._direction) + ' radians'

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return repr('TensionVector({})'.format(self._corresponding_edge))
