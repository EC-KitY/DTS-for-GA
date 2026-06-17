import logging
import os

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection

from .common import build_dts_operator
from ..caching_evaluator import CachingEvaluator
from ..elitist_breeder import ElitistBreeder
from ..logging_utils import FileLogger


def make_selection(
    kind,
    population_size,
    vocab_size,
    dts_cfg=None,
    device="cpu",
    custom_reward_function=None,
):
    """Return a selection operator: 'dts' (learned) or 'tournament' (baseline)."""
    if kind == "dts":
        kwargs = {"custom_reward_function": custom_reward_function}
        if dts_cfg is not None:
            kwargs.update(
                latent_dim=dts_cfg.latent_dim,
                dim_feedforward=dts_cfg.dim_feedforward,
                n_layers=dts_cfg.n_layers,
                n_heads=dts_cfg.n_heads,
                tournament_size=dts_cfg.tournament_size,
                learning_rate=dts_cfg.learning_rate,
                final_lr=dts_cfg.final_lr,
                train_every_n_gens=dts_cfg.train_every_n_gens,
                epsilon_greedy=dts_cfg.epsilon_greedy,
                epsilon_greedy_decay=dts_cfg.epsilon_greedy_decay,
                min_epsilon=dts_cfg.min_epsilon,
            )
        return build_dts_operator(
            population_size=population_size,
            vocab_size=vocab_size,
            device=device,
            **kwargs,
        )
    elif kind == "tournament":
        t = dts_cfg.tournament_size if dts_cfg is not None else 5
        return TournamentSelection(tournament_size=t, higher_is_better=True)
    raise ValueError(f"unknown selection kind: {kind}")


def run_one(
    label,
    creator,
    evaluator,
    operators,
    selection,
    *,
    population_size,
    generations,
    elitism,
    output_path=None,
    diversity_fn=None,
    use_cache=True,
    quiet=False,
    max_workers=1,
):
    """Run a single evolution and return a result dict.

    All problems here are framed as MAXIMIZATION (fitness is negated cost), so
    higher_is_better=True throughout — which is also what DTS assumes.
    """
    if use_cache:
        evaluator = CachingEvaluator(evaluator)

    stats = [FileLogger(output_path, diversity_fn=diversity_fn)]
    if not quiet:
        stats.append(BestAverageWorstStatistics())

    algo = SimpleEvolution(
        Subpopulation(
            creators=creator,
            population_size=population_size,
            evaluator=evaluator,
            higher_is_better=True,
            elitism_rate=elitism / population_size,
            operators_sequence=operators,
            selection_methods=[(selection, 1)],
        ),
        breeder=ElitistBreeder(),
        max_workers=max_workers,
        max_generation=generations,
        statistics=stats,
    )
    algo.evolve()

    best = algo.best_of_run_.get_pure_fitness()
    result = {"label": label, "best_fitness": best, "generations": generations}
    if use_cache:
        result["cache_stats"] = evaluator.cache_stats()
    json_stats = stats[0]
    saved = json_stats.save(extra={"label": label})
    if saved:
        result["results_file"] = saved
    return result


def resolve_instances(data_subdir, instance_arg, default_list):
    """Resolve which instance file paths to run.

    instance_arg: a filename (in the bundled data dir), an absolute path, the
    string 'all', or None (-> use default_list).
    """
    from ..problems import DATA_DIR

    base = os.path.join(DATA_DIR, data_subdir)

    def to_path(name):
        return name if os.path.isabs(name) else os.path.join(base, name)

    if instance_arg in (None, "all"):
        names = default_list if instance_arg is None else sorted(os.listdir(base))
        return [(n, to_path(n)) for n in names]
    return [(os.path.basename(instance_arg), to_path(instance_arg))]


def configure_logging(quiet):
    logging.basicConfig(level=logging.INFO)
    if quiet:
        logging.disable(logging.INFO)
