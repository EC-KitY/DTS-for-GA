import numpy as np


def _average_pairwise_hamming(population: np.ndarray) -> float:
    pop = np.asarray(population)
    n, length = pop.shape
    if n < 2:
        return 0.0
    diffs = (pop[:, None, :] != pop[None, :, :]).sum(axis=2)
    triu = np.triu_indices(n, k=1)
    return float((diffs[triu] / length).mean())


def set_cover_diversity(population: np.ndarray) -> float:
    pop = np.asarray(population, dtype=np.int32)
    n = pop.shape[0]
    if n < 2:
        return 0.0
    intersections = pop @ pop.T
    counts = pop.sum(axis=1, dtype=np.int32)
    unions = counts[:, None] + counts[None, :] - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    distances = 1.0 - similarities
    triu = np.triu_indices(n, k=1)
    return float(distances[triu].mean())


def _canonicalize_coloring(population: np.ndarray) -> np.ndarray:
    pop = np.asarray(population, dtype=np.int64)
    out = np.empty_like(pop)
    for i in range(pop.shape[0]):
        mapping, nxt = {}, 0
        for j, color in enumerate(pop[i].tolist()):
            if color not in mapping:
                mapping[color] = nxt
                nxt += 1
            out[i, j] = mapping[color]
    return out


def graph_coloring_diversity(population: np.ndarray) -> float:
    pop = np.asarray(population, dtype=np.int64)
    if pop.ndim != 2 or len(pop) < 2:
        return 0.0
    return _average_pairwise_hamming(_canonicalize_coloring(pop))


def tsp_edge_diversity(population: np.ndarray) -> float:
    pop = np.asarray(population, dtype=np.int64)
    n, tour_length = pop.shape
    if n < 2:
        return 0.0
    n_cities = int(pop.max()) + 1
    edge_sets = []
    for tour in pop:
        nxt = np.roll(tour, -1)
        a = np.minimum(tour, nxt)
        b = np.maximum(tour, nxt)
        edge_sets.append(set((a * n_cities + b).tolist()))
    total, pairs = 0.0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            shared = len(edge_sets[i] & edge_sets[j])
            total += 1.0 - shared / tour_length
            pairs += 1
    return total / pairs
