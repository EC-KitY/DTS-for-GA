import os

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation

from ..caching_evaluator import CachingEvaluator
from ..config import DTSConfig, TSPConfig
from ..elitist_breeder import ElitistBreeder
from ..experiments.runner_utils import make_selection
from ..logging_utils import FileLogger
from ..problems import (
    DATA_DIR,
    PermutationVectorCreator,
    RSMMutation,
    SCXCrossover,
    TSPEvaluator,
)
from ..problems.diversity import tsp_edge_diversity

# --------------------------------------------------------------------------- #
# 1. PARAMETERS
# --------------------------------------------------------------------------- #
cfg = TSPConfig()
dts_cfg = DTSConfig()

INSTANCE = "berlin52.tsp"  # file under problems/data/tsp/
SELECTION = "dts"  # "dts" (learned) or "tournament" (baseline)
GENERATIONS = 1000  # cfg.generations is 1000 in the paper
POPULATION_SIZE = cfg.population_size  # 100
CROSSOVER_PROB = cfg.crossover_prob  # 0.5
MUTATION_PROB = cfg.mutation_prob  # 0.5 (operator-level RSM mutation)
ELITISM = cfg.elitism  # 2
DEVICE = "cpu"  # "cpu" or "cuda"
OUTPUT_PATH = os.path.join("runs", "tsp", INSTANCE, SELECTION, "run_0.json")

# --------------------------------------------------------------------------- #
# 2. EVALUATOR  — load the instance, wrap it in the cross-generation fitness cache
# --------------------------------------------------------------------------- #
instance_path = os.path.join(DATA_DIR, "tsp", INSTANCE)
evaluator = TSPEvaluator(instance_path)
num_cities = evaluator.num_cities
distance_matrix = evaluator.distance_matrix
evaluator = CachingEvaluator(evaluator)

# --------------------------------------------------------------------------- #
# 3. ENCODING + OPERATORS  — permutation (tour order); SCX crossover + RSM mutation
# --------------------------------------------------------------------------- #
creator = PermutationVectorCreator(num_cities)
operators = [
    SCXCrossover(distance_matrix, probability=CROSSOVER_PROB),
    RSMMutation(probability=MUTATION_PROB),
]

# --------------------------------------------------------------------------- #
# 4. SELECTION  — DTS
# --------------------------------------------------------------------------- #
selection = make_selection(
    SELECTION, POPULATION_SIZE, vocab_size=num_cities, dts_cfg=dts_cfg, device=DEVICE
)

# --------------------------------------------------------------------------- #
# 5. ASSEMBLE THE GA  — elitism-protecting breeder + per-generation JSON logger
# --------------------------------------------------------------------------- #
logger = FileLogger(OUTPUT_PATH, diversity_fn=tsp_edge_diversity)
algo = SimpleEvolution(
    Subpopulation(
        creators=creator,
        population_size=POPULATION_SIZE,
        evaluator=evaluator,
        higher_is_better=True,
        elitism_rate=ELITISM / POPULATION_SIZE,
        operators_sequence=operators,
        selection_methods=[(selection, 1)],
    ),
    breeder=ElitistBreeder(),
    max_generation=GENERATIONS,
    statistics=[logger],
)

# --------------------------------------------------------------------------- #
# 6. RUN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    algo.evolve()
    best = algo.best_of_run_.get_pure_fitness()
    print(f"best tour length = {-best:.0f}")
    saved = logger.save(extra={"label": f"tsp/{INSTANCE}/{SELECTION}"})
    print(f"metrics saved to: {saved}")
