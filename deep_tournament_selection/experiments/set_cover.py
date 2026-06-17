import argparse
import os

from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
)

from ..config import SetCoverConfig, DTSConfig
from ..problems import SetCoverEvaluator, VectorUniformCrossover
from ..problems.diversity import set_cover_diversity
from .runner_utils import make_selection, run_one, resolve_instances, configure_logging


def main():
    cfg, dts = SetCoverConfig(), DTSConfig()
    p = argparse.ArgumentParser(description="Set Cover with Deep Tournament Selection")
    p.add_argument(
        "--instance", default=None, help="filename in data/set_cover, a path, or 'all'"
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

    for name, path in resolve_instances("set_cover", args.instance, cfg.instances):
        sizing = SetCoverEvaluator(path, penalty=cfg.penalty, weighted=cfg.weighted)
        n_columns = sizing.n_columns
        print(
            f"\n=== Set Cover {name}  (columns={n_columns}, rows={sizing.n_rows})  "
            f"selection={args.selection} ==="
        )
        for run in range(args.runs):
            evaluator = SetCoverEvaluator(
                path, penalty=cfg.penalty, weighted=cfg.weighted
            )
            creator = GABitStringVectorCreator(length=n_columns)
            operators = [
                VectorUniformCrossover(probability=args.crossover_prob),
                BitStringVectorNFlipMutation(
                    probability=args.mutation_prob,
                    probability_for_each=args.flip_mutation_prob,
                    n=n_columns,
                ),
            ]
            selection = make_selection(
                args.selection,
                args.population_size,
                vocab_size=2,
                dts_cfg=dts,
                device=args.device,
            )
            out = os.path.join(
                args.output, "set_cover", name, args.selection, f"run_{run}.json"
            )
            res = run_one(
                f"sc/{name}/{args.selection}/run{run}",
                creator,
                evaluator,
                operators,
                selection,
                population_size=args.population_size,
                generations=args.generations,
                elitism=cfg.elitism,
                output_path=out,
                quiet=args.quiet,
                diversity_fn=None if args.no_diversity else set_cover_diversity,
            )
            print(
                f"  run {run}: best fitness = {res['best_fitness']:.0f}  "
                f"(higher is better: -cost - {cfg.penalty:.0f}*uncovered)"
            )


if __name__ == "__main__":
    main()
