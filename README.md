# Deep Tournament Selection (DTS) for Genetic Algorithms

Implementation of the paper **"Deep Tournament Selection for Genetic Algorithms"**
(Eliad Shem-Tov, Ron Edri, Achiya Elyasaf — Ben-Gurion University of the Negev).
📄 **Paper:** _link coming soon_.

<p align="center">
  <img src="images/dts_arch.png" alt="Deep Tournament Selection architecture" width="800">
</p>

## Installation

```bash
pip install eckity-dts
```

## Public API

```python
from eckity_dts import CachingEvaluator, DeepTournamentSelection, DTSPolicy
```

## Constructing DTS

The policy combines a population encoder with a pointer network:

```python
from eckity_dts import DeepTournamentSelection, DTSPolicy
from deep_tournament_selection.selection.population_to_vec_transformer import (
    PopulationToVecTransformer,
)
from deep_tournament_selection.selection.self_attention_pointer import (
    SelfAttentionPointer,
)

population_size = 100
vocab_size = 2  # maximum gene value + 1

encoder = PopulationToVecTransformer(
    vocab_size=vocab_size,
    emb_dim=32,
    latent_dim=32,
    n_heads=4,
    n_layers=2,
    dim_feedforward=256,
    max_pointers=population_size,
)

pointer = SelfAttentionPointer(
    pointer_len=population_size,
    d_model=32,
)

policy = DTSPolicy(
    pop_to_vec_transformer=encoder,
    pointer_transformer=pointer,
    device="cpu",
    train_every_n_gens=10,
    learning_rate=2e-3,
    final_lr=1e-3,
    epsilon_greedy=1.0,
    epsilon_greedy_decay=0.999,
    min_epsilon=0.2,
)

dts = DeepTournamentSelection(policy)
```

Use it as the EC-KitY selection method:

```python
selection_methods=[dts]
```

Important parameters:

- `population_size` determines the rank-embedding and pointer-table capacities.
- `vocab_size` is the maximum integer gene value plus one.
- `train_every_n_gens` controls how often accumulated trajectories train the policy.
- `epsilon_greedy` and its decay control teacher-forced tournament selection versus learned selection.
- `custom_reward_function`, when supplied, receives current fitness, previous fitness, and population arrays.
- `device` can be `"cpu"` or `"cuda"` when a compatible PyTorch installation is available.

## Fitness caching

EC-KitY may evaluate unchanged vectors again across generations. `CachingEvaluator` avoids recomputing fitness for vectors it has already seen:

```python
from eckity_dts import CachingEvaluator

evaluator = CachingEvaluator(MyEvaluator())
print(evaluator.cache_stats())
```

## Compatibility

- Python 3.9 or newer
- EC-KitY 0.4.x
- NumPy 2.0.2 or newer
- SciPy 1.13.0 or newer
- PyTorch 2.7.1 or newer
- overrides 7.7.0 or newer

These bounds are compatible with `eckity-dnc`, `eckity-bert-ga`, and `eckity-bert-gp`. None of the operator packages depends directly on another operator package.

## Research repository

The repository contains the full paper experiments for Graph Coloring, Set Cover, and TSP, together with benchmark instances, notebook-style runners, and result figures. These research resources are not included in the `eckity-dts` wheel.

For repository development:

```bash
uv sync --extra dev --resolution lowest-direct
uv run pytest
```

Experiment entry points remain available from a source checkout:

```bash
python -m deep_tournament_selection.experiments.graph_coloring --instance queen8_12.col.txt --generations 200
python -m deep_tournament_selection.experiments.set_cover --instance scp41.txt --generations 200
python -m deep_tournament_selection.experiments.tsp --instance att48.tsp --generations 200
python run_experiments.py --selection both --runs 3 --generations 500
```

Results are written under `runs/`. Paper figures are stored under `figures/`, and the architecture diagram is under `images/`.

## Development and release

```bash
uv run pytest
uv run ruff check .
uv build
```

Release preparation and manual PyPI upload commands are documented in [`RELEASING.md`](RELEASING.md).

## License

This project is licensed under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
