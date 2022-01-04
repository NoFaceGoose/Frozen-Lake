def position_to_index(x, y, num_cols):
    """
        Converts position(row index & column index) in the grid to index in the flat representation of the grid.
        Formula: (number of columns * row index) + column index

        Example: 2D array: [[1, 2],
                            [3, 4]]
                 Flat array: [1, 2, 3, 4]
                 position of 3 is (1, 0), index of will be ((2 * 1) + 0) = 2

    :param x: row index of the grid
    :param y: column index of the grid
    :param num_cols: Number of columns in the 2D array/grid
    :return: index in the flat representation of the grid
    """

    return (num_cols * x) + y


def index_to_position(val, num_cols):
    """
        Converts index in the flat representation of the grid to position(row index & column index) in the grid.
        Formula: row index = (index / number of columns)
                 column index = (index % number of columns)

        Example: Flat array: [1, 2, 3, 4]
                 2D array: [[1, 2],
                            [3, 4]]
                 index of 3 is 2, position of 3 will be ((2 / 2), (2 % 2)) = (1, 0)

    :param val: index in the flat representation of the grid
    :param num_cols: Number of columns in the 2D array/grid
    :return: row index, column index as a tuple
    """

    return int(val / num_cols), val % num_cols
