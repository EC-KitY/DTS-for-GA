from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.elitism_selection import ElitismSelection


class ElitistBreeder(SimpleBreeder):
    def apply_breed(self, population):
        for subpopulation in population.sub_populations:
            nextgen_population = []

            num_elites = subpopulation.n_elite
            if num_elites > 0:
                ElitismSelection(
                    num_elites=num_elites,
                    higher_is_better=subpopulation.higher_is_better,
                ).apply_operator((subpopulation.individuals, nextgen_population))

            selection = subpopulation.get_selection_methods()[0][0]
            nextgen_population = selection.select(
                subpopulation.individuals, nextgen_population
            )
            self.selected_individuals = nextgen_population

            elites = nextgen_population[:num_elites]
            offspring = self._apply_operators(
                subpopulation.get_operators_sequence(),
                nextgen_population[num_elites:],
            )
            subpopulation.individuals = elites + offspring
