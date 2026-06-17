import argparse
import csv
import os
import time

from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
    IntVectorOnePointMutation,
)

from deep_tournament_selection.config import (
    DTSConfig,
    GraphColoringConfig,
    SetCoverConfig,
    TSPConfig,
)
from deep_tournament_selection.problems import (
    DATA_DIR,
    GraphColoringEvaluator,
    PermutationVectorCreator,
    RSMMutation,
    SCXCrossover,
    SetCoverEvaluator,
    TSPEvaluator,
    VectorUniformCrossover,
)
from deep_tournament_selection.problems.diversity import (
    graph_coloring_diversity,
    set_cover_diversity,
    tsp_edge_diversity,
)
from deep_tournament_selection.experiments.runner_utils import (
    configure_logging,
    make_selection,
    run_one,
)


# --------------------------------------------------------------------------- #
# Per-problem builders: each returns (evaluator, creator, operators, vocab_size,
# size_label) for a single fresh run on the given instance.
# --------------------------------------------------------------------------- #
def build_tsp(path, cfg, cross_prob, mut_prob):
    ev = TSPEvaluator(path)
    n = ev.num_cities
    creator = PermutationVectorCreator(n)
    operators = [
        SCXCrossover(ev.distance_matrix, probability=cross_prob),
        RSMMutation(probability=mut_prob),
    ]
    return ev, creator, operators, n, f"cities={n}"


def build_graph_coloring(path, cfg, cross_prob, mut_prob):
    ev = GraphColoringEvaluator(path, penalty=cfg.penalty)
    n = ev.n_nodes
    max_colors = n - cfg.colors_margin
    creator = GAIntVectorCreator(length=n, bounds=(0, max_colors))
    operators = [
        VectorUniformCrossover(probability=cross_prob),
        IntVectorOnePointMutation(
            probability=mut_prob, probability_for_each=cfg.flip_mutation_prob
        ),
    ]
    return ev, creator, operators, max_colors + 1, f"nodes={n}"


def build_set_cover(path, cfg, cross_prob, mut_prob):
    ev = SetCoverEvaluator(path, penalty=cfg.penalty, weighted=cfg.weighted)
    n = ev.n_columns
    creator = GABitStringVectorCreator(length=n)
    operators = [
        VectorUniformCrossover(probability=cross_prob),
        BitStringVectorNFlipMutation(
            probability=mut_prob, probability_for_each=cfg.flip_mutation_prob, n=n
        ),
    ]
    return ev, creator, operators, 2, f"columns={n}"


PROBLEMS = {
    "tsp": dict(
        cfg=TSPConfig(),
        data="tsp",
        build=build_tsp,
        score=lambda f: -f,
        score_name="tour_length",
        diversity=tsp_edge_diversity,
    ),
    "graph_coloring": dict(
        cfg=GraphColoringConfig(),
        data="graph_coloring",
        build=build_graph_coloring,
        score=lambda f: f,
        score_name="fitness",
        diversity=graph_coloring_diversity,
    ),
    "set_cover": dict(
        cfg=SetCoverConfig(),
        data="set_cover",
        build=build_set_cover,
        score=lambda f: f,
        score_name="fitness",
        diversity=set_cover_diversity,
    ),
}


def resolve_instances(data_subdir, requested, default_list):
    base = os.path.join(DATA_DIR, data_subdir)
    if not requested:
        names = default_list
    elif requested == ["all"]:
        names = sorted(os.listdir(base))
    else:
        names = requested
    out = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(base, name)
        if os.path.exists(path):
            out.append((os.path.basename(name), path))
        else:
            print(f"  [skip] instance not found: {name}")
    return out


def main():
    p = argparse.ArgumentParser(description="Run the DTS experiments.")
    p.add_argument(
        "--problems",
        nargs="+",
        default=list(PROBLEMS),
        choices=list(PROBLEMS),
        help="which problems to run",
    )
    p.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="instance filenames, or 'all'; default = each problem's config list",
    )
    p.add_argument("--selection", choices=["dts", "tournament", "both"], default="dts")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument(
        "--generations",
        type=int,
        default=None,
        help="override generations (default: per-problem config)",
    )
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--output", default="runs")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--quick",
        action="store_true",
        help="tiny fast settings on one small instance per problem (smoke test)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="full paper protocol: per-problem config instances/runs/generations",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--no-diversity",
        action="store_true",
        help="skip population-diversity logging (faster, esp. on large TSP)",
    )
    args = p.parse_args()
    configure_logging(quiet=True if args.quick else args.quiet)

    quick_instance = {
        "tsp": ["att48.tsp"],
        "graph_coloring": ["queen8_12.col.txt"],
        "set_cover": ["scp41.txt"],
    }
    dts_cfg = DTSConfig()
    selections = ["dts", "tournament"] if args.selection == "both" else [args.selection]

    summary_rows = []
    for prob_name in args.problems:
        spec = PROBLEMS[prob_name]
        cfg = spec["cfg"]

        instances_arg = args.instances
        if args.quick and instances_arg is None:
            instances_arg = quick_instance[prob_name]
        instances = resolve_instances(spec["data"], instances_arg, cfg.instances)

        generations = (
            10
            if args.quick
            else (
                cfg.generations
                if args.full or args.generations is None
                else args.generations
            )
        )
        if args.generations is not None and not args.full:
            generations = args.generations
        population_size = (
            20 if args.quick else (args.population_size or cfg.population_size)
        )
        runs = 1 if args.quick else (cfg.runs if args.full else args.runs)

        for inst_name, inst_path in instances:
            for sel in selections:
                for run in range(runs):
                    ev, creator, operators, vocab_size, size_label = spec["build"](
                        inst_path, cfg, cfg.crossover_prob, cfg.mutation_prob
                    )
                    selection = make_selection(
                        sel,
                        population_size,
                        vocab_size,
                        dts_cfg=dts_cfg,
                        device=args.device,
                    )
                    out = os.path.join(
                        args.output, prob_name, inst_name, sel, f"run_{run}.json"
                    )
                    t0 = time.time()
                    res = run_one(
                        f"{prob_name}/{inst_name}/{sel}/run{run}",
                        creator,
                        ev,
                        operators,
                        selection,
                        population_size=population_size,
                        generations=generations,
                        elitism=cfg.elitism,
                        output_path=out,
                        quiet=True,
                        diversity_fn=None if args.no_diversity else spec["diversity"],
                    )
                    dt = time.time() - t0
                    score = spec["score"](res["best_fitness"])
                    row = dict(
                        problem=prob_name,
                        instance=inst_name,
                        size=size_label,
                        selection=sel,
                        run=run,
                        best_fitness=res["best_fitness"],
                        **{spec["score_name"]: score},
                        seconds=round(dt, 1),
                    )
                    summary_rows.append(row)
                    print(
                        f"[{prob_name:14s} {inst_name:22s} {sel:10s} run{run}] "
                        f"{spec['score_name']}={score:.1f}  ({dt:.1f}s)"
                    )

    # Write summary CSV
    if summary_rows:
        os.makedirs(args.output, exist_ok=True)
        csv_path = os.path.join(args.output, "summary.csv")
        keys = [
            "problem",
            "instance",
            "size",
            "selection",
            "run",
            "best_fitness",
            "tour_length",
            "fitness",
            "seconds",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary written to {csv_path}  ({len(summary_rows)} runs)")


if __name__ == "__main__":
    main()
