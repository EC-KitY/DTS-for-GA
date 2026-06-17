import os

import numpy as np
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
)
from eckity.subpopulation import Subpopulation

from ..caching_evaluator import CachingEvaluator
from ..config import DTSConfig
from ..elitist_breeder import ElitistBreeder
from ..experiments.runner_utils import make_selection
from ..logging_utils import FileLogger
from ..problems import VectorUniformCrossover

# --------------------------------------------------------------------------- #
# General GA parameters — edit freely
# --------------------------------------------------------------------------- #
dts_cfg = DTSConfig()
GENERATIONS = 100
POPULATION_SIZE = 100
DEVICE = "cpu"  # "cpu" or "cuda"
OUTPUT_PATH = os.path.join("runs", "custom", "run_0.json")


# =========================================================================== #
# (A) YOUR EVALUATOR (the fitness function; higher = better)
# =========================================================================== #
class OneMaxEvaluator(SimpleIndividualEvaluator):
    """Dummy problem: maximize the number of 1s in the bit string."""

    def evaluate_individual(self, individual):
        return float(np.sum(individual.vector))


evaluator = OneMaxEvaluator()

# =========================================================================== #
# YOUR ENCODING: the creator + genetic operators
# VOCAB_SIZE: max gene value + 1 (here bits 0/1 -> 2)
#
# The creator defines the genome; the operators must match its type. DTS works
# with any integer/bit vector or permutation encoding.
# =========================================================================== #
GENOME_LENGTH = 50
VOCAB_SIZE = 2
creator = GABitStringVectorCreator(length=GENOME_LENGTH)
operators = [
    VectorUniformCrossover(probability=0.5),
    BitStringVectorNFlipMutation(
        probability=0.5, probability_for_each=0.1, n=GENOME_LENGTH
    ),
]

# =========================================================================== #
# (D) OPTIONAL  —  change how DTS is TRAINED with a custom reward
#
# Leave REWARD_FN = None to use the default reward. To customize, point it at a
# function f(cur_gen_fitness, prev_gen_fitness, population) -> float
# =========================================================================== #
REWARD_FN = None

# =========================================================================== #
# DTS + GA assembly, identical for any problem
# =========================================================================== #
evaluator = CachingEvaluator(evaluator)  # cross-generation fitness cache

selection = make_selection(
    "dts",
    POPULATION_SIZE,
    vocab_size=VOCAB_SIZE,
    dts_cfg=dts_cfg,
    device=DEVICE,
    custom_reward_function=REWARD_FN,
)

logger = FileLogger(OUTPUT_PATH)
algo = SimpleEvolution(
    Subpopulation(
        creators=creator,
        population_size=POPULATION_SIZE,
        evaluator=evaluator,
        higher_is_better=True,
        elitism_rate=2 / POPULATION_SIZE,
        operators_sequence=operators,
        selection_methods=[(selection, 1)],
    ),
    breeder=ElitistBreeder(),
    max_generation=GENERATIONS,
    statistics=[logger],
)

if __name__ == "__main__":
    algo.evolve()
    best = algo.best_of_run_.get_pure_fitness()
    print(f"best fitness = {best:.0f}  (OneMax optimum = {GENOME_LENGTH})")
    saved = logger.save(extra={"label": "custom"})
    print(f"metrics saved to: {saved}")
