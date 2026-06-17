import re

import numpy as np
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


def load_data(input_path):
    """Load one or more SCP cases. Returns a list of case dicts with keys
    m, n, costs, row_cover (row_cover[i] = 1-indexed columns covering row i)."""
    with open(input_path, "r") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    first = blocks[0].split()
    if len(first) == 1:
        try:
            int(first[0])
            blocks = blocks[1:]
        except Exception:
            pass
    return [_parse_single_case(b) for b in blocks]


def _parse_single_case(block):
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty test case block.")
    header = lines[0].split()
    m, n = int(header[0]), int(header[1])
    remaining = lines[1:]

    if len(remaining) == n:
        costs, col_rows = [], []
        for j in range(n):
            tokens = remaining[j].split()
            cost, count = int(tokens[0]), int(tokens[1])
            col_rows.append(list(map(int, tokens[2 : 2 + count])))
            costs.append(cost)
        row_cover = [[] for _ in range(m)]
        for j in range(n):
            for r in col_rows[j]:
                row_cover[r - 1].append(j + 1)
        return {"m": m, "n": n, "costs": costs, "row_cover": row_cover}

    elif len(remaining) == m:
        return {
            "m": m,
            "n": n,
            "costs": [1] * n,
            "row_cover": [list(map(int, remaining[i].split())) for i in range(m)],
        }

    else:
        cost_tokens, idx = [], 0
        while idx < len(remaining) and len(cost_tokens) < n:
            cost_tokens.extend(remaining[idx].split())
            idx += 1
        costs = list(map(int, cost_tokens[:n]))
        row_tokens = []
        for line in remaining[idx:]:
            row_tokens.extend(line.split())
        t, row_cover = 0, []
        for _ in range(m):
            k = int(row_tokens[t])
            t += 1
            row_cover.append(list(map(int, row_tokens[t : t + k])))
            t += k
        return {"m": m, "n": n, "costs": costs, "row_cover": row_cover}


class SetCoverEvaluator(SimpleIndividualEvaluator):
    """Fitness = -(total cost + penalty * #uncovered rows). Maximize."""

    def __init__(
        self, path_to_instance, penalty=100.0, weighted=False, case_index=0, events=None
    ):
        super().__init__(events=events)
        case = load_data(path_to_instance)[case_index]
        self.n_columns = case["n"]
        self.n_rows = case["m"]
        self.costs = np.array(case["costs"])
        self.penalty = penalty
        self.weighted = weighted
        self.row_cover = [set(c - 1 for c in cols) for cols in case["row_cover"]]

    def evaluate_individual(self, individual):
        vec = np.asarray(individual.vector)
        selected = set(np.flatnonzero(vec == 1).tolist())
        total_cost = (
            float(self.costs[vec == 1].sum()) if self.weighted else float(vec.sum())
        )
        uncovered = sum(1 for cover in self.row_cover if not (cover & selected))
        return float(-(total_cost + self.penalty * uncovered))
