import json
import os
import time

import numpy as np
from eckity.statistics.statistics import Statistics


class FileLogger(Statistics):
    BASE_KEYS = ("mean", "std", "median", "max", "min", "time")

    def __init__(
        self,
        output_path=None,
        save_every_n_generations=50,
        diversity_fn=None,
        format_string=None,
    ):
        super().__init__(format_string or "")
        self.output_path = output_path
        self.save_every_n_generations = save_every_n_generations
        self.diversity_fn = diversity_fn
        keys = self.BASE_KEYS + (("population_diversity",) if diversity_fn else ())
        self.generation_metrics = {key: [] for key in keys}
        self._last_time = time.time()

    def write_statistics(self, sender, data_dict):
        sub_pop = data_dict["population"].sub_populations[0]
        fitness_values = np.array(
            [ind.get_pure_fitness() for ind in sub_pop.individuals], dtype=float
        )
        self.generation_metrics["mean"].append(float(np.mean(fitness_values)))
        self.generation_metrics["std"].append(float(np.std(fitness_values)))
        self.generation_metrics["median"].append(float(np.median(fitness_values)))
        self.generation_metrics["max"].append(float(np.max(fitness_values)))
        self.generation_metrics["min"].append(float(np.min(fitness_values)))

        if self.diversity_fn is not None:
            population = np.array([ind.vector for ind in sub_pop.individuals])
            self.generation_metrics["population_diversity"].append(
                float(self.diversity_fn(population))
            )

        now = time.time()
        self.generation_metrics["time"].append(now - self._last_time)
        self._last_time = now

        if (
            self.output_path
            and self.save_every_n_generations
            and data_dict["generation_num"] % self.save_every_n_generations == 0
        ):
            self.save()

    def save(self, extra=None):
        if self.output_path is None:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        payload = dict(self.generation_metrics)
        if extra:
            payload.update(extra)
        with open(self.output_path, "w") as f:
            json.dump(payload, f, indent=2)
        return self.output_path
