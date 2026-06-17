from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DTSConfig:
    """Hyperparameters for the Deep Tournament Selection operator (paper setting)."""

    tournament_size: int = 5
    latent_dim: int = 32
    dim_feedforward: int = 256
    n_layers: int = 2
    n_heads: int = 4
    learning_rate: float = 2e-3
    final_lr: float = 1e-3
    train_every_n_gens: int = 10
    epsilon_greedy: float = 1.0
    epsilon_greedy_decay: float = 0.999
    min_epsilon: float = 0.2


POPULATION_SIZE = 100
ELITISM = 2
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.5
FLIP_MUTATION_PROB = 0.1
RUNS = 15


@dataclass(frozen=True)
class TSPConfig:
    instances: List[str] = field(
        default_factory=lambda: [
            "att48.tsp",
            "berlin52.tsp",
            "st70.tsp",
            "lin105.tsp",
            "pr107.tsp",
            "pcb442.tsp",
            "d1291.tsp",
        ]
    )
    population_size: int = POPULATION_SIZE
    generations: int = 1000
    crossover_prob: float = CROSSOVER_PROB
    mutation_prob: float = MUTATION_PROB
    elitism: int = ELITISM
    runs: int = RUNS


@dataclass(frozen=True)
class GraphColoringConfig:
    instances: List[str] = field(
        default_factory=lambda: [
            "instances_games120.col.txt",
            "instances_myciel7.col.txt",
            "miles1000.col.txt",
            "miles1500.col.txt",
            "mulsol.i.2.col",
            "queen8_12.col.txt",
            "zeroin.i.1.col",
            "zeroin.i.2.col",
        ]
    )
    population_size: int = POPULATION_SIZE
    generations: int = 6000
    crossover_prob: float = CROSSOVER_PROB
    mutation_prob: float = MUTATION_PROB
    flip_mutation_prob: float = FLIP_MUTATION_PROB
    penalty: float = 100.0
    colors_margin: int = 10
    elitism: int = ELITISM
    runs: int = RUNS


@dataclass(frozen=True)
class SetCoverConfig:
    instances: List[str] = field(
        default_factory=lambda: [
            "scp41.txt",
            "scp51.txt",
            "scp52.txt",
            "scp53.txt",
            "scp54.txt",
            "scp56.txt",
            "scp57.txt",
            "scp64.txt",
            "scp65.txt",
        ]
    )
    population_size: int = POPULATION_SIZE
    generations: int = 6000
    crossover_prob: float = CROSSOVER_PROB
    mutation_prob: float = MUTATION_PROB
    flip_mutation_prob: float = FLIP_MUTATION_PROB
    penalty: float = 100.0
    weighted: bool = False
    elitism: int = ELITISM
    runs: int = RUNS
