import argparse
import os

from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_operators.mutations.vector_random_mutation import (
    IntVectorOnePointMutation,
)

from ..config import GraphColoringConfig, DTSConfig
from ..problems import GraphColoringEvaluator, VectorUniformCrossover
from ..problems.diversity import graph_coloring_diversity
from .runner_utils import make_selection, run_one, resolve_instances, configure_logging


def main():
    cfg, dts = GraphColoringConfig(), DTSConfig()
    p = argparse.ArgumentParser(
        description="Graph Coloring with Deep Tournament Selection"
    )
    p.add_argument(
        "--instance",
        default=None,
        help="filename in data/graph_coloring, a path, or 'all'",
    )
    p.add_argument("--selection", choices=["dts", "tournament"], default="dts")
    p.add_argument("--population-size", type=int, default=cfg.population_size)
    p.add_argument("--generations", type=int, default=cfg.generations)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--crossover-prob", type=float, default=cfg.crossover_prob)
    p.add_argument("--mutation-prob", type=float, default=cfg.mutation_prob)
    p.add_argument("--flip-mutation-prob", type=float, default=cfg.flip_mutation_prob)
    p.add_argument("--output", default="runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--no-diversity", action="store_true", help="skip population-diversity logging"
    )
    args = p.parse_args()
    configure_logging(args.quiet)

    for name, path in resolve_instances("graph_coloring", args.instance, cfg.instances):
        sizing = GraphColoringEvaluator(path, penalty=cfg.penalty)
        n_nodes = sizing.n_nodes
        max_colors = n_nodes - cfg.colors_margin
        print(
            f"\n=== Graph Coloring {name}  (nodes={n_nodes}, max_colors={max_colors})  "
            f"selection={args.selection} ==="
        )
        for run in range(args.runs):
            evaluator = GraphColoringEvaluator(path, penalty=cfg.penalty)
            creator = GAIntVectorCreator(length=n_nodes, bounds=(0, max_colors))
            operators = [
                VectorUniformCrossover(probability=args.crossover_prob),
                IntVectorOnePointMutation(
                    probability=args.mutation_prob,
                    probability_for_each=args.flip_mutation_prob,
                ),
            ]
            selection = make_selection(
                args.selection,
                args.population_size,
                vocab_size=max_colors + 1,
                dts_cfg=dts,
                device=args.device,
            )
            out = os.path.join(
                args.output, "graph_coloring", name, args.selection, f"run_{run}.json"
            )
            res = run_one(
                f"gc/{name}/{args.selection}/run{run}",
                creator,
                evaluator,
                operators,
                selection,
                population_size=args.population_size,
                generations=args.generations,
                elitism=cfg.elitism,
                output_path=out,
                quiet=args.quiet,
                diversity_fn=None if args.no_diversity else graph_coloring_diversity,
            )
            print(
                f"  run {run}: best fitness = {res['best_fitness']:.0f}  "
                f"(higher is better: -colors - {cfg.penalty:.0f}*conflicts)"
            )


if __name__ == "__main__":
    main()
