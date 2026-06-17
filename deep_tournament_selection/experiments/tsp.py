import argparse
import os

from ..config import TSPConfig, DTSConfig
from ..problems import TSPEvaluator, PermutationVectorCreator, SCXCrossover, RSMMutation
from ..problems.diversity import tsp_edge_diversity
from .runner_utils import make_selection, run_one, resolve_instances, configure_logging


def main():
    cfg, dts = TSPConfig(), DTSConfig()
    p = argparse.ArgumentParser(description="TSP with Deep Tournament Selection")
    p.add_argument(
        "--instance", default=None, help="filename in data/tsp, a path, or 'all'"
    )
    p.add_argument("--selection", choices=["dts", "tournament"], default="dts")
    p.add_argument("--population-size", type=int, default=cfg.population_size)
    p.add_argument("--generations", type=int, default=cfg.generations)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--crossover-prob", type=float, default=cfg.crossover_prob)
    p.add_argument("--mutation-prob", type=float, default=cfg.mutation_prob)
    p.add_argument("--output", default="runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--no-diversity", action="store_true", help="skip population-diversity logging"
    )
    args = p.parse_args()
    configure_logging(args.quiet)

    for name, path in resolve_instances("tsp", args.instance, cfg.instances):
        sizing = TSPEvaluator(path)
        num_cities = sizing.num_cities
        print(
            f"\n=== TSP {name}  (cities={num_cities})  selection={args.selection} ==="
        )
        for run in range(args.runs):
            evaluator = TSPEvaluator(path)
            operators = [
                SCXCrossover(
                    evaluator.distance_matrix, probability=args.crossover_prob
                ),
                RSMMutation(probability=args.mutation_prob),
            ]
            selection = make_selection(
                args.selection,
                args.population_size,
                vocab_size=num_cities,
                dts_cfg=dts,
                device=args.device,
            )
            out = os.path.join(
                args.output, "tsp", name, args.selection, f"run_{run}.json"
            )
            res = run_one(
                f"tsp/{name}/{args.selection}/run{run}",
                PermutationVectorCreator(num_cities),
                evaluator,
                operators,
                selection,
                population_size=args.population_size,
                generations=args.generations,
                elitism=cfg.elitism,
                output_path=out,
                quiet=args.quiet,
                diversity_fn=None if args.no_diversity else tsp_edge_diversity,
            )
            print(f"  run {run}: best tour length = {-res['best_fitness']:.0f}")


if __name__ == "__main__":
    main()
