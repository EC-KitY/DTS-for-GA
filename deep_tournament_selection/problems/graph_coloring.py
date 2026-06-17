import numpy as np
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


def parse_graph_file(file_path):
    """Parse a DIMACS .col file -> (edges_information, n_nodes, n_edges).

    edges_information: {1-indexed node -> list of 1-indexed neighbors}.
    """
    with open(file_path, "r") as f:
        raw_lines = f.readlines()

    p_lines = [line for line in raw_lines if line.startswith("p")]
    assert len(p_lines) == 1
    n_nodes, n_edges = p_lines[0].split()[2:4]
    n_nodes, n_edges = int(n_nodes), int(n_edges)

    edges_information = {i + 1: [] for i in range(n_nodes)}
    for line in raw_lines:
        if not line.startswith("e"):
            continue
        v1, v2 = (int(x) for x in line.split()[1:3])
        edges_information[v1].append(v2)
        edges_information[v2].append(v1)
    return edges_information, n_nodes, n_edges


class GraphColoringEvaluator(SimpleIndividualEvaluator):
    """Fitness = -(#colors + penalty * #conflicting-nodes). Maximize."""

    def __init__(self, path_to_instance, penalty=100, events=None):
        super().__init__(events=events)
        edges_information, n_nodes, n_edges = parse_graph_file(path_to_instance)
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.penalty = penalty
        u, v = [], []
        for node, neighbors in edges_information.items():
            for nb in neighbors:
                if node < nb:
                    u.append(node - 1)
                    v.append(nb - 1)
        self.edge_u = np.array(u, dtype=np.int64)
        self.edge_v = np.array(v, dtype=np.int64)

    def evaluate_individual(self, individual):
        colors = np.asarray(individual.vector)
        num_colors = len(np.unique(colors))
        conflict_edges = colors[self.edge_u] == colors[self.edge_v]
        conflicting_nodes = np.unique(
            np.concatenate([self.edge_u[conflict_edges], self.edge_v[conflict_edges]])
        )
        return float(-num_colors - self.penalty * len(conflicting_nodes))
