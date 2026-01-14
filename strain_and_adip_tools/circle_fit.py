import math

import numpy as np
from scipy import optimize


def distance(point_1, point_2):
    """ calculates the euclidian distance between two points

    :param point_1:
    :param point_2:
    :return:
    """
    x1 = point_1[0]
    x2 = point_2[0]
    y1 = point_1[1]
    y2 = point_2[1]
    # Calculating distance
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_circle(x1, y1, x2, y2, x3, y3):
    """finds center and radius of a circle passing through three points

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param x3:
    :param y3:
    :return:
    """
    points = np.asarray([[2 * x1, 2 * y1, 1], [2 * x2, 2 * y2, 1], [2 * x3, 2 * y3, 1]])
    radii = np.asarray([[-(x1 ** 2 + y1 ** 2)], [-(x2 ** 2 + y2 ** 2)], [-(x3 ** 2 + y3 ** 2)]])
    try:
        solution = np.linalg.solve(points, radii)
        a = solution[0][0]
        b = solution[1][0]
        xc = -a
        yc = -b
        r = distance((x1, y1), (xc, yc))
    except np.linalg.LinAlgError:
        # a singular matrix occurs when the three points are approximately collinear, so set these values very large
        # (The small arc of a very large circle approximates a straight line)
        xc = 10000000000
        yc = 10000000000
        r = math.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)

    return xc, yc, r


def fit(x, y, start_point, end_point):
    """ function that does the fitting

    :param x:
    :param y:
    :param start_point:
    :param end_point:
    :return:
    """
    # initial guess is just a point in the middle of the arc
    initial_guess = np.asarray([x[1], y[1]])

    # goal is to find the third point that determines the arc
    # the third point much be located in the bounding box of all provided points with a buffer of 20 pixels (?)
    lb = np.asarray([np.min(x), np.min(y)]) - 20
    ub = np.asarray([np.max(x), np.max(y)]) + 20
    bounds = optimize.Bounds(lb, ub)
    third_point = optimize.minimize(fun=cost, x0=initial_guess, args=(x, y, start_point, end_point), bounds=bounds)
    third_point = third_point.x
    xc, yc, radius = find_circle(start_point[0], start_point[1], third_point[0], third_point[1], end_point[0],
                                 end_point[1])
    return xc, yc, radius


def cost(third_point, x, y, start_point, end_point):
    """ cost function needs to minimized (sum of squared differences between actual radius and fit radius
    (least-squares)

    :return:
    """
    xc, yc, radius = find_circle(start_point[0], start_point[1], third_point[0], third_point[1], end_point[0],
                                 end_point[1])
    individual_radii = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    residuals = individual_radii - radius
    least_squares = np.sum(np.square(residuals))

    return least_squares
