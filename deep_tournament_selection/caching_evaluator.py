from collections import OrderedDict
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class CachingEvaluator(SimpleIndividualEvaluator):
    def __init__(self, inner: SimpleIndividualEvaluator, max_size=None, events=None):
        super().__init__(events=events)
        self.inner = inner
        self.max_size = max_size
        self.fitness_cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def evaluate_individual(self, individual):
        key = tuple(individual.vector)
        if key in self.fitness_cache:
            self.hits += 1
            self.fitness_cache.move_to_end(key)
            return self.fitness_cache[key]

        self.misses += 1
        value = self.inner.evaluate_individual(individual)
        self.fitness_cache[key] = value
        if self.max_size is not None and len(self.fitness_cache) > self.max_size:
            self.fitness_cache.popitem(last=False)
        return value

    def cache_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.fitness_cache),
            "hit_rate": hit_rate,
        }
