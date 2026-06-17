from __future__ import annotations

import numpy as np
import torch


def get_ranks_from_fitness_values(fitness_values: np.ndarray):
    order = np.argsort(-fitness_values, axis=1)
    ranks = np.argsort(order, axis=1)
    return ranks


def get_ranking_based_rewards(fitness_values: np.ndarray, gamma=1.25):
    fitness_values = fitness_values.reshape(1, -1)
    ranks = get_ranks_from_fitness_values(fitness_values)
    ranks_weights = 1 / (ranks + 1) ** gamma
    rewards = ranks_weights * fitness_values
    rewards = rewards.sum(axis=1)
    return rewards


def get_exponential_ranking_based_rewards(fitness_values: np.ndarray, alpha=0.2):
    fitness_values = fitness_values.reshape(1, -1)
    ranks = get_ranks_from_fitness_values(fitness_values)
    ranks_weights = np.exp(-alpha * ranks)
    rewards = ranks_weights * fitness_values
    rewards = rewards.sum(axis=1)
    return rewards


def get_average_pairwise_hamming_distance(population: np.ndarray) -> float:
    population = np.asarray(population)

    if population.ndim != 2 or len(population) < 2:
        return 0.0

    diffs = population[:, None, :] != population[None, :, :]
    pairwise_dist = diffs.sum(axis=2)
    triu_indices = np.triu_indices(len(population), k=1)
    return float(pairwise_dist[triu_indices].mean() / population.shape[1])


def get_reward_from_fitness_scores(
    cur_gen_fitness_values: np.ndarray,
    prev_gen_fitness_values: np.ndarray,
    population: np.ndarray | None = None,
    top_k_to_consider: int = 5,
    best_weight: float = 0.25,
    top_k_weight: float = 1.0,
    mean_weight: float = 0.25,
    diversity_weight: float = 0.05,
    eps: float = 1e-8,
):
    cur_gen_fitness_values = np.asarray(cur_gen_fitness_values, dtype=np.float64)
    prev_gen_fitness_values = np.asarray(prev_gen_fitness_values, dtype=np.float64)

    k = min(
        top_k_to_consider, len(cur_gen_fitness_values), len(prev_gen_fitness_values)
    )

    def normalized_delta(cur_value: float, prev_value: float) -> float:
        scale = max(abs(prev_value), eps)
        return (cur_value - prev_value) / scale

    cur_top_k_mean = np.partition(cur_gen_fitness_values, -k)[-k:].mean()
    prev_top_k_mean = np.partition(prev_gen_fitness_values, -k)[-k:].mean()

    cur_best = cur_gen_fitness_values.max()
    prev_best = prev_gen_fitness_values.max()

    cur_mean = cur_gen_fitness_values.mean()
    prev_mean = prev_gen_fitness_values.mean()

    diversity_reward = 0.0
    if population is not None:
        diversity_reward = get_average_pairwise_hamming_distance(population)

    reward = (
        top_k_weight * normalized_delta(cur_top_k_mean, prev_top_k_mean)
        + best_weight * normalized_delta(cur_best, prev_best)
        + mean_weight * normalized_delta(cur_mean, prev_mean)
        + diversity_weight * diversity_reward
    )
    return float(reward)


def get_trajectory_probability_from_log_probs(log_probs, selected_population_indices):
    """
    calculate total probability for the trajectory by summing the log probabilities over the selected indices
    """
    trajectory_prob = (
        torch.gather(log_probs, 2, selected_population_indices.unsqueeze(-1))
        .squeeze(-1)
        .sum(dim=-1)
    )
    return trajectory_prob
