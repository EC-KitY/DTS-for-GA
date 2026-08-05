import ast
from importlib.metadata import version
from pathlib import Path

from eckity.genetic_operators import SelectionMethod, TournamentSelection

from deep_tournament_selection import DeepTournamentSelection
from deep_tournament_selection import elitist_breeder as breeder_module
from deep_tournament_selection.elitist_breeder import ElitistBreeder
from deep_tournament_selection.experiments.runner_utils import make_selection


ROOT = Path(__file__).parents[1]
SOURCE_FILES = [
    ROOT / "deep_tournament_selection" / "elitist_breeder.py",
    ROOT / "deep_tournament_selection" / "experiments" / "runner_utils.py",
    ROOT / "deep_tournament_selection" / "runners" / "custom_runner.py",
    ROOT / "deep_tournament_selection" / "runners" / "graph_coloring.py",
    ROOT / "deep_tournament_selection" / "runners" / "set_cover.py",
    ROOT / "deep_tournament_selection" / "runners" / "tsp.py",
]


class FakeSelection:
    def select(self, source, destination):
        destination.append("offspring")
        return destination


class FakeSubpopulation:
    def __init__(self, num_elites=0):
        self.individuals = ["candidate"]
        self.n_elite = num_elites
        self.selection = FakeSelection()

    def get_selection_methods(self):
        return [self.selection]

    def get_operators_sequence(self):
        return []


class FakePopulation:
    def __init__(self, subpopulation):
        self.sub_populations = [subpopulation]


def test_release_metadata_and_public_imports():
    from eckity_dts import CachingEvaluator, DTSPolicy

    assert version("eckity") == "0.4.2"
    assert version("eckity-dts") == "0.1.1"
    assert all(item is not None for item in (CachingEvaluator, DTSPolicy))


def test_deep_selection_accepts_legacy_direction_keyword():
    selection = DeepTournamentSelection(policy=object(), higher_is_better=False)
    assert isinstance(selection, SelectionMethod)
    assert not hasattr(selection, "higher_is_better")


def test_tournament_factory_uses_new_constructor():
    selection = make_selection("tournament", population_size=4, vocab_size=2)
    assert isinstance(selection, TournamentSelection)


def test_custom_breeder_reads_selection_method_directly():
    subpopulation = FakeSubpopulation()
    breeder = ElitistBreeder()
    breeder._apply_operators = lambda operators, individuals: individuals

    breeder.apply_breed(FakePopulation(subpopulation))

    assert subpopulation.individuals == ["offspring"]


def test_custom_breeder_constructs_elitism_without_direction(monkeypatch):
    calls = []

    class FakeElitismSelection:
        def __init__(self, num_elites):
            calls.append(num_elites)

        def apply_operator(self, payload):
            payload[1].append("elite")
            return payload[1]

    monkeypatch.setattr(breeder_module, "ElitismSelection", FakeElitismSelection)
    subpopulation = FakeSubpopulation(num_elites=1)
    breeder = ElitistBreeder()
    breeder._apply_operators = lambda operators, individuals: individuals

    breeder.apply_breed(FakePopulation(subpopulation))

    assert calls == [1]
    assert subpopulation.individuals == ["elite", "offspring"]


def test_sources_do_not_use_legacy_selection_tuples_or_direction_keywords():
    for path in SOURCE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            assert not (
                call_name in {"TournamentSelection", "ElitismSelection"}
                and any(keyword.arg == "higher_is_better" for keyword in node.keywords)
            )
            for keyword in node.keywords:
                if keyword.arg == "selection_methods" and isinstance(
                    keyword.value, ast.List
                ):
                    assert all(
                        not isinstance(item, ast.Tuple) for item in keyword.value.elts
                    )
