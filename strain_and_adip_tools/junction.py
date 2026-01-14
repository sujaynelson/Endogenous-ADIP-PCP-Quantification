import itertools

import matplotlib.pyplot as plt


class Junction:
    id_iter = itertools.count()

    def __init__(self, coordinates, cells_set):
        self._coordinates = coordinates
        self._edge_labels = set()
        self._label = next(Junction.id_iter)
        self._cell_labels = cells_set
        self.x_unit_vectors_dict = {}  # each dictionary entry is edge label: x_unit_vector_component
        self.y_unit_vectors_dict = {}  # each dictionary entry is edge label: y_unit_vector_component
        

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        if len(coordinates) == 2:
            self._coordinates = coordinates
        else:
            raise ValueError('coordinates should not exceed length of 2. The length of coordinates was: {}'.format(
                len(coordinates)))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def edges(self):
        """set of labels of edges connected to this node"""

        return self._edge_labels

    def add_edge(self, edge_label):
        """Adds edge label to set of edge labels"""
        self._edge_labels.add(edge_label)

    def remove_edge(self, edge_label):
        """Remove an edge and tension vector connected to this node

        :param self:
        :param edge_label:
        """

        try:
            self._edge_labels.remove(edge_label)
        except ValueError:
            raise ValueError("{} is not connected to this Junction".format(edge_label))

    @property
    def tension_vectors(self):
        """ returns list of Tension vectors connected to this node"""

        tension_vectors = []
        for edge in self._edge_labels:
            tension_vectors.append(edge.corresponding_tension_vector)

        return tension_vectors

    @property
    def degree(self):
        return len(self._cell_labels)

    def plot(self, label=False):
        plt.scatter(self.x, self.y, c='r', s=10)
        if label:
            plt.text(self.coordinates[0], self.coordinates[1], str(self._label), color='black', fontsize=2.5,
                     horizontalalignment='center', verticalalignment='center')

    def plot_unit_vectors(self):
        # for edge_label in self.x_unit_vectors_dict:
        #     x=self.x_unit_vectors_dict[edge_label]
        #     plt.plot([self.x,self.x+10*x],[self.y,self.y], lw=0.75)
        # for edge_label in self.y_unit_vectors_dict:
        #     y=self.y_unit_vectors_dict[edge_label]
        #     plt.plot([self.x,self.x],[self.y,self.y+10*y], lw=0.75)

        for edge_label in self.x_unit_vectors_dict:
            x = self.x_unit_vectors_dict[edge_label]
            y = self.y_unit_vectors_dict[edge_label]
            plt.plot([self.x, self.x + 10 * x], [self.y, self.y + 10 * y], lw=0.75)

    @property
    def cell_labels(self):
        return self._cell_labels

    def __eq__(self, other):
        return self._coordinates == other.coordinates

    def __str__(self):
        return str(self._coordinates)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return repr('Junction({})'.format(self._coordinates))
