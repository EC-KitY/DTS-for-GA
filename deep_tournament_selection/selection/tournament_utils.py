from copy import deepcopy

import numpy as np


def get_fit(ind, fitness_dict):
    """Return the cached fitness for an individual (keyed by tuple(ind))."""
    code = tuple(ind)
    if code in fitness_dict:
        return fitness_dict[code]


def save_fitness(ind, val, fitness_dict):
    """Cache the fitness ``val`` for an individual (keyed by tuple(ind))."""
    fitness_dict[tuple(ind)] = val


def choose_from_competition(competition, fitness_dict, return_index=False):
    """Return the winner of a single tournament (argmax fitness).

    :param competition: list/array of individuals taking part in the tournament
    :param fitness_dict: fitness cache keyed by tuple(ind)
    :param return_index: if True, return the winner's index within ``competition``
    """
    fitnesses = [fitness_dict[tuple(ind)] for ind in competition]
    if return_index:
        return np.argmax(fitnesses)
    else:
        return np.copy(competition[np.argmax(fitnesses)])


def tournament_selection(
    individuals, how_many_to_select, tournament_size, fitness_dict, return_index=False
):
    """Classic tournament selection (used by DTS as its teacher-forcing baseline).

    :param individuals: population (numpy array of gene-vectors)
    :param how_many_to_select: number of winners to produce
    :param tournament_size: number of competitors per tournament
    :param fitness_dict: fitness cache keyed by tuple(ind)
    :param return_index: if True, return (selected_population_indexes, all_tournament_indexes)
                         instead of the selected individuals themselves.
    """
    tournament_indexes: np.ndarray = np.random.randint(
        0, len(individuals), size=(how_many_to_select, tournament_size)
    )
    tournaments = [
        individuals[tournament_index] for tournament_index in tournament_indexes
    ]
    selected_individuals = [
        choose_from_competition(tournament, fitness_dict, return_index=return_index)
        for tournament in tournaments
    ]

    if return_index:
        selected_population_indexes = tournament_indexes[
            np.arange(tournament_indexes.shape[0]), selected_individuals
        ]
        return selected_population_indexes, tournament_indexes
    else:
        return deepcopy(np.array(selected_individuals))
