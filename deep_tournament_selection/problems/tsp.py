from math import sqrt, ceil

import numpy as np
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


def load_tsplib_tsp(path):
    """Parse a TSPLIB file -> (distance_matrix, coordinates). Supports EUC_2D and ATT."""
    coords, dimension, edge_weight_type = [], None, None
    with open(path, "r") as f:
        lines = f.readlines()

    reading = False
    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1])
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line == "NODE_COORD_SECTION":
            reading = True
        elif line == "EOF":
            break
        elif reading:
            _, x, y = line.split()
            coords.append((float(x), float(y)))

    assert len(coords) == dimension
    coords = np.array(coords)
    n = dimension
    dist = np.zeros((n, n), dtype=int)
    if edge_weight_type == "EUC_2D":
        for i in range(n):
            for j in range(n):
                dx, dy = coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]
                dist[i, j] = int(round(sqrt(dx * dx + dy * dy)))
    elif edge_weight_type == "ATT":
        for i in range(n):
            for j in range(n):
                dx, dy = coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]
                dist[i, j] = int(ceil(sqrt((dx * dx + dy * dy) / 10.0)))
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE {edge_weight_type} not supported")
    return dist, coords


class TSPEvaluator(SimpleIndividualEvaluator):
    """Fitness = -(total tour length). Maximize (higher_is_better=True)."""

    def __init__(self, path_to_instance, events=None):
        super().__init__(events=events)
        self.distance_matrix, self.coordinates = load_tsplib_tsp(path_to_instance)
        self.num_cities = len(self.distance_matrix)

    def evaluate_individual(self, individual):
        tour = np.asarray(individual.vector)
        next_tour = np.roll(tour, -1)
        return float(-self.distance_matrix[tour, next_tour].sum())
