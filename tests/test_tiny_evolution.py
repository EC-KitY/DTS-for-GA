import random

import numpy as np
import torch
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
)
from eckity.subpopulation import Subpopulation

from deep_tournament_selection.elitist_breeder import ElitistBreeder
from deep_tournament_selection.experiments.common import build_dts_operator
from deep_tournament_selection.problems import VectorUniformCrossover


class OneMaxEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        return float(sum(individual.vector))


def test_tiny_bit_vector_evolves_and_trains_dts():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    selection = build_dts_operator(
        population_size=4,
        vocab_size=2,
        latent_dim=4,
        emb_dim=4,
        n_heads=2,
        n_layers=1,
        dim_feedforward=4,
        tournament_size=2,
        device="cpu",
        learning_rate=1e-3,
        final_lr=None,
        train_every_n_gens=2,
        epsilon_greedy=1.0,
        epsilon_greedy_decay=1.0,
        min_epsilon=1.0,
    )
    evolution = SimpleEvolution(
        population=Subpopulation(
            creators=GAVectorCreator(length=4, bounds=(0, 1)),
            population_size=4,
            evaluator=OneMaxEvaluator(),
            higher_is_better=True,
            elitism_rate=0.5,
            operators_sequence=[
                VectorUniformCrossover(probability=1.0),
                BitStringVectorNFlipMutation(
                    probability=1.0,
                    probability_for_each=0.5,
                ),
            ],
            selection_methods=[selection],
        ),
        breeder=ElitistBreeder(),
        max_workers=1,
        max_generation=4,
        random_seed=7,
    )

    evolution.evolve()

    individuals = evolution.population.sub_populations[0].individuals
    best_fitness = float(evolution.best_of_run_.get_pure_fitness())
    assert evolution.generation_num == 4
    assert selection.generation_index == 4
    assert len(individuals) == 4
    assert 0.0 <= best_fitness <= 4.0
    assert all(
        len(individual.vector) == 4 and set(individual.vector) <= {0, 1}
        for individual in individuals
    )
    assert selection.policy.optimizer.state
    assert all(
        torch.isfinite(parameter).all()
        for parameter in selection.policy.all_parameters
    )
