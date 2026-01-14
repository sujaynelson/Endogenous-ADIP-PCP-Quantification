# This class represents a node
class Node:

    # Initialize the class
    def __init__(self, position: (), parent: ()):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node
        self.f = 0  # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.position == other.position

    # Sort nodes
    def __lt__(self, other):
        return self.f < other.f

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))


# Breadth-first search (BFS)
def breadth_first_search(maze, start, end):
    # Create lists for open nodes and closed nodes
    open = []
    closed = []

    # Create a start node and an goal node
    start_node = Node(start, None)
    goal_node = Node(end, None)

    # Add the start node
    open.append(start_node)

    # Loop until the open list is empty
    while len(open) > 0:

        # Get the first node (FIFO)
        current_node = open.pop(0)

        # Add the current node to the closed list
        closed.append(current_node)

        # Check if we have reached the goal, return the path
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.append(start)
            # Return reversed path
            return path[::-1]

        # Unzip the current node position
        row, col = current_node.position

        # find position of all the neighbors
        neighbors = []
        if row > 0:
            north = (row - 1, col)
            neighbors.append(north)
        if col < len(maze[0]) - 1:
            east = (row, col + 1)
            neighbors.append(east)
        if row < len(maze) - 1:
            south = (row + 1, col)
            neighbors.append(south)
        if col > 0:
            west = (row, col - 1)
            neighbors.append(west)

        # Loop neighbors
        for neighbor in neighbors:

            # Get value from map
            map_value = maze[neighbor[0]][neighbor[1]]

            # Check if the node is a wall
            if map_value == 0:
                continue

            # Create a neighbor node
            neighbor = Node(neighbor, current_node)

            # Check if the neighbor is in the closed list
            if neighbor in closed:
                continue

            # Everything is green, add the node if it not is in open
            if neighbor not in open:
                open.append(neighbor)

    # Return None, no path is found
    return None

