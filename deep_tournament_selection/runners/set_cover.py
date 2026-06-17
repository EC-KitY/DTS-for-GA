import os

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
)
from eckity.subpopulation import Subpopulation

from ..caching_evaluator import CachingEvaluator
from ..config import DTSConfig, SetCoverConfig
from ..elitist_breeder import ElitistBreeder
from ..experiments.runner_utils import make_selection
from ..logging_utils import FileLogger
from ..problems import DATA_DIR, SetCoverEvaluator, VectorUniformCrossover
from ..problems.diversity import set_cover_diversity

# --------------------------------------------------------------------------- #
# 1. PARAMETERS
# --------------------------------------------------------------------------- #
cfg = SetCoverConfig()
dts_cfg = DTSConfig()

INSTANCE = "scp65.txt"  # file under problems/data/set_cover/
SELECTION = "dts"  # "dts" (learned) or "tournament" (baseline)
GENERATIONS = 1000  # cfg.generations is 6000 in the paper
POPULATION_SIZE = cfg.population_size  # 100
CROSSOVER_PROB = cfg.crossover_prob  # 0.5
MUTATION_PROB = cfg.mutation_prob  # 0.5 (operator-level)
FLIP_MUTATION_PROB = cfg.flip_mutation_prob  # 0.1 (per-gene)
ELITISM = cfg.elitism  # 2
DEVICE = "cpu"  # "cpu" or "cuda"
OUTPUT_PATH = os.path.join("runs", "set_cover", INSTANCE, SELECTION, "run_0.json")

# --------------------------------------------------------------------------- #
# 2. EVALUATOR  — load the instance, wrap it in the cross-generation fitness cache
# --------------------------------------------------------------------------- #
instance_path = os.path.join(DATA_DIR, "set_cover", INSTANCE)
evaluator = SetCoverEvaluator(instance_path, penalty=cfg.penalty, weighted=cfg.weighted)
n_columns = evaluator.n_columns
evaluator = CachingEvaluator(evaluator)

# --------------------------------------------------------------------------- #
# 3. ENCODING + OPERATORS  — bit vector (one bit per subset)
# --------------------------------------------------------------------------- #
creator = GABitStringVectorCreator(length=n_columns)
operators = [
    VectorUniformCrossover(probability=CROSSOVER_PROB),
    BitStringVectorNFlipMutation(
        probability=MUTATION_PROB,
        probability_for_each=FLIP_MUTATION_PROB,
        n=n_columns,
    ),
]

# --------------------------------------------------------------------------- #
# 4. SELECTION  — DTS
# --------------------------------------------------------------------------- #
selection = make_selection(
    SELECTION, POPULATION_SIZE, vocab_size=2, dts_cfg=dts_cfg, device=DEVICE
)

# --------------------------------------------------------------------------- #
# 5. ASSEMBLE THE GA  — elitism-protecting breeder + per-generation JSON logger
# --------------------------------------------------------------------------- #
logger = FileLogger(OUTPUT_PATH, diversity_fn=set_cover_diversity)
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
    print(f"best fitness = {best:.0f}  (-cost - {cfg.penalty:.0f} * uncovered)")
    saved = logger.save(extra={"label": f"set_cover/{INSTANCE}/{SELECTION}"})
    print(f"metrics saved to: {saved}")
