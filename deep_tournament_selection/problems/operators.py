import random

import numpy as np
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_operators.genetic_operator import GeneticOperator


class VectorUniformCrossover(GeneticOperator):
    """Uniform crossover: swap each gene between the two parents with prob 0.5.

    Mirrors the original `_uniform_crossover` but as a standard arity-2 EC-KitY
    operator (operates on a pair, like VectorKPointsCrossover).
    """

    def __init__(self, probability=1.0, arity=2, swap_probability=0.5, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
        self.swap_probability = swap_probability

    def apply(self, individuals):
        v1, v2 = individuals[0].vector, individuals[1].vector
        for i in range(len(v1)):
            if random.random() < self.swap_probability:
                v1[i], v2[i] = v2[i], v1[i]
        self.applied_individuals = individuals
        return individuals


class PermutationVectorCreator(GAIntVectorCreator):
    """Creates IntVector individuals initialized as random permutations of 0..length-1."""

    def __init__(self, length=1, events=None):
        super().__init__(length=length, bounds=(0, length - 1), events=events)

    def create_vector(self, individual):
        individual.set_vector(list(np.random.permutation(self.length)))


class SCXCrossover(GeneticOperator):
    """Sequential Constructive Crossover (SCX) for the TSP.

    Builds a child by, at each step, choosing the better (shorter-edge) of the
    next unused cities that follow the current city in each parent; falls back to
    the nearest unused city. Produces two children (one per parent ordering).
    Ported from ga_auxiliary.py's `_scx_crossover_impl` / `generate_scx_children`.
    """

    def __init__(self, distance_matrix, probability=1.0, arity=2, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
        self.distance_matrix = np.ascontiguousarray(distance_matrix)

    def apply(self, individuals):
        p1 = np.asarray(individuals[0].vector, dtype=np.int64)
        p2 = np.asarray(individuals[1].vector, dtype=np.int64)
        child1 = self._scx(p1, p2)
        child2 = self._scx(p2, p1)
        individuals[0].set_vector(list(child1))
        individuals[1].set_vector(list(child2))
        self.applied_individuals = individuals
        return individuals

    def _scx(self, parent1, parent2):
        dm = self.distance_matrix
        n = len(parent1)
        n_cities = dm.shape[0]
        child = np.empty(n, dtype=np.int64)
        used = np.zeros(n_cities, dtype=bool)
        pos1 = np.empty(n_cities, dtype=np.int64)
        pos2 = np.empty(n_cities, dtype=np.int64)
        for idx in range(n):
            pos1[parent1[idx]] = idx
            pos2[parent2[idx]] = idx

        current = parent1[0]
        child[0] = current
        used[current] = True
        for i in range(1, n):
            c1 = self._next_unused(parent1, pos1, current, used)
            c2 = self._next_unused(parent2, pos2, current, used)
            if c1 == -1 and c2 == -1:
                nxt = self._nearest_unused(current, used)
            elif c1 == -1:
                nxt = c2
            elif c2 == -1:
                nxt = c1
            else:
                nxt = c1 if dm[current, c1] <= dm[current, c2] else c2
            child[i] = nxt
            used[nxt] = True
            current = nxt
        return child

    @staticmethod
    def _next_unused(parent, positions, current_city, used):
        n = len(parent)
        cur = positions[current_city]
        for offset in range(1, n + 1):
            candidate = parent[(cur + offset) % n]
            if not used[candidate]:
                return candidate
        return -1

    def _nearest_unused(self, current_city, used):
        dm = self.distance_matrix
        best_city, best_dist = -1, np.inf
        for city in range(dm.shape[0]):
            if not used[city] and dm[current_city, city] < best_dist:
                best_dist = dm[current_city, city]
                best_city = city
        return best_city


class RSMMutation(GeneticOperator):
    """Reverse Sequence Mutation (RSM): reverse a random sub-segment of the tour.

    Permutation-preserving. Ported from ga_auxiliary.py's `reverse_sequence_mutation`.
    """

    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(probability=probability, arity=arity, events=events)

    def apply(self, individuals):
        for ind in individuals:
            vec = ind.vector
            n = len(vec)
            i, j = sorted(random.sample(range(n), 2))
            vec[i : j + 1] = vec[i : j + 1][::-1]
        self.applied_individuals = individuals
        return individuals
