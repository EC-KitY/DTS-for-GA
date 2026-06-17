import os

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_operators.mutations.vector_random_mutation import (
    IntVectorOnePointMutation,
)
from eckity.subpopulation import Subpopulation

from ..caching_evaluator import CachingEvaluator
from ..config import DTSConfig, GraphColoringConfig
from ..elitist_breeder import ElitistBreeder
from ..experiments.runner_utils import make_selection
from ..logging_utils import FileLogger
from ..problems import DATA_DIR, GraphColoringEvaluator, VectorUniformCrossover
from ..problems.diversity import graph_coloring_diversity

# --------------------------------------------------------------------------- #
# 1. PARAMETERS
# --------------------------------------------------------------------------- #
cfg = GraphColoringConfig()
dts_cfg = DTSConfig()

INSTANCE = "instances_myciel7.col.txt"  # file under problems/data/graph_coloring/
SELECTION = "dts"  # "dts" (learned) or "tournament" (baseline)
GENERATIONS = 2000  # cfg.generations is 6000 in the paper
POPULATION_SIZE = cfg.population_size  # 100
CROSSOVER_PROB = cfg.crossover_prob  # 0.5
MUTATION_PROB = cfg.mutation_prob  # 0.5 (operator-level)
FLIP_MUTATION_PROB = cfg.flip_mutation_prob  # 0.1 (per-gene)
ELITISM = cfg.elitism  # 2
DEVICE = "cpu"  # "cpu" or "cuda"
OUTPUT_PATH = os.path.join("runs", "graph_coloring", INSTANCE, SELECTION, "run_0.json")

# --------------------------------------------------------------------------- #
# 2. EVALUATOR  — load the instance, wrap it in the cross-generation fitness cache
# --------------------------------------------------------------------------- #
instance_path = os.path.join(DATA_DIR, "graph_coloring", INSTANCE)
evaluator = GraphColoringEvaluator(instance_path, penalty=cfg.penalty)
n_nodes = evaluator.n_nodes
max_colors = n_nodes - cfg.colors_margin
evaluator = CachingEvaluator(evaluator)

# --------------------------------------------------------------------------- #
# 3. ENCODING + OPERATORS  — integer vector (one color per vertex)
# --------------------------------------------------------------------------- #
creator = GAIntVectorCreator(length=n_nodes, bounds=(0, max_colors))
operators = [
    VectorUniformCrossover(probability=CROSSOVER_PROB),
    IntVectorOnePointMutation(
        probability=MUTATION_PROB, probability_for_each=FLIP_MUTATION_PROB
    ),
]

# --------------------------------------------------------------------------- #
# 4. SELECTION  — DTS
# --------------------------------------------------------------------------- #
selection = make_selection(
    SELECTION,
    POPULATION_SIZE,
    vocab_size=max_colors + 1,
    dts_cfg=dts_cfg,
    device=DEVICE,
)

# --------------------------------------------------------------------------- #
# 5. ASSEMBLE THE GA
# --------------------------------------------------------------------------- #
logger = FileLogger(OUTPUT_PATH, diversity_fn=graph_coloring_diversity)
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
    print(f"best fitness = {best:.0f}  (-colors - {cfg.penalty:.0f} * conflicts)")
    saved = logger.save(extra={"label": f"graph_coloring/{INSTANCE}/{SELECTION}"})
    print(f"metrics saved to: {saved}")
