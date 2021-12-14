from unittest import TestCase

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from utils.constraints import FIELD_NAMES_CONSTRAINTS, FIELD_NAMES_SOLUTIONS
from utils.generation import (calcule_all, calcule_fitness, crossover,
                              init_generation, penalize, rand_crossover,
                              ranking, selection)
from utils.plot import plot
from utils.transformer import Transformer

from tests.data_for_testes import *

COLORS = list(colors.TABLEAU_COLORS.values())



class TestAGBasic(TestCase):
    def test_create_ag(self):
        population = init_generation(n_population, variations)
        # import ipdb; ipdb.set_trace()
        assert population.index.__len__() == n_population

    def test_calcule_all(self):
        population = init_generation(n_population, variations)
        transformer = Transformer(constraints=constraints, tables=tables)
        res = calcule_all(population, transformer)
        assert res.index.__len__() == n_population
        # import ipdb; ipdb.set_trace()

    def test_penalize(self):
        population = init_generation(n_population, variations)
        transformer = Transformer(constraints=constraints, tables=tables)
        res = calcule_all(population, transformer)
        all = pd.concat([population, res], axis=1)
        penalise_res = penalize(all, variations + [(0, 1900), (0, 590)])
        # import ipdb; ipdb.set_trace()

    def test_ranking(self):
        population = init_generation(
            n_population=n_population,
            variations=variations
        )
        transformer = Transformer(constraints=constraints, tables=tables)
        res = calcule_all(population, transformer)
        all = pd.concat([population, res], axis=1)
        penalise_res = penalize(all, variations + [(0, 1900), (0, 590)])
        ranking_res = ranking(penalise_res)

        # plot(penalise_res, ranking_res)
        # plot(res, ranking_res)
    
        # plt.show()
        for_test = pd.DataFrame()
        for_test["PerdasT"] = [1, 2, 5]
        for_test["Mativa"] = [2, 4, 1]
        # import ipdb; ipdb.set_trace()
        x = ranking(for_test)

        all_with_ranking = pd.concat([all, ranking_res], axis=1)

    def test_ranking_error(self):
        x = """
        0  1.347670  1.470260  1.582003  6.359731  0.502712  3.580778  1.139337  1832.160885  548.017914
        1  1.304333  1.454479  1.562767  6.538517  0.505836  3.551132  1.152717  1808.545483  561.457646
        2  1.389956  1.503746  1.558400  6.839850  0.482279  3.416636  1.133621  1878.749760  520.287074
        3  1.345110  1.426099  1.525679  6.476215  0.492902  3.485161  1.147620  1817.926655  561.114516
        4  1.227814  1.407598  1.578368  6.360522  0.519284  3.518718  1.104476  1771.653203  573.572737
        """.strip().rstrip().split("\n")

        y = [k.strip().rstrip().split("  ")[1:] for k in x]


        # import ipdb; ipdb.set_trace()
        df = pd.DataFrame(y, columns=FIELD_NAMES_CONSTRAINTS + FIELD_NAMES_SOLUTIONS)

        df = df.apply(lambda c: np.float32(c), axis=0)

        x = ranking(df)

    def test_calcule_fitness(self):
        population = init_generation(
            n_population=n_population,
            variations=variations
        )

        transformer = Transformer(constraints=constraints, tables=tables)
        res = calcule_all(population, transformer)
        # import ipdb; ipdb.set_trace()
        all = pd.concat([population, res], axis=1)
        penalise_res = penalize(all, variations + [(0, 1900), (0, 590)])

        ranking_res = ranking(penalise_res)
        # plot(penalise_res, ranking_res)

        # plt.show()
        fitness = calcule_fitness(
            penalise_res,
            ranking_res,
            variations_solutions=[(0, 1900), (0, 590)],
            niche_ray=.1,
            alfa=2,
            const_mult=3
        )
        # import ipdb; ipdb.set_trace()

class TestAG(TestCase):
    def setUp (self):
        population = init_generation(
            n_population=n_population,
            variations=variations
        )

        transformer = Transformer(constraints=constraints, tables=tables)
        res = calcule_all(population, transformer)
        all = pd.concat([population, res], axis=1)
        penalise_res = penalize(all, variations + [(0, 1900), (0, 590)])

        ranking_res = ranking(penalise_res)
        
        fitness = calcule_fitness(
            penalise_res,
            ranking_res,
            variations_solutions=[(0, 1900), (0, 590)],
            niche_ray=.1,
            alfa=2,
            const_mult=3
        )

        self.population = all
        self.fitness = fitness
        self.transformer = transformer


    def test_crossover(self):

        res = crossover (
            self.population,
            fitness=self.fitness,
            crossover_probability=.32,
            disturbance_rate=1.7,
            frac_crossover=.9,
        )
        # import ipdb; ipdb.set_trace()
        after_cal = calcule_all(res[FIELD_NAMES_CONSTRAINTS], self.transformer)
        res.update(after_cal)

        all = pd.concat([res, self.population], axis=0)
        all.index = range(len(all))
        penalise_res = penalize(all, variations + [(0, 1900), (0, 590)])
        all.update(penalise_res)
        ranks = ranking(penalise_res)

        # plot(penalise_res, ranks, title="all data")
        # plt.show()
    
    def __test_rand_crossover(self):
        res = rand_crossover (
            population=self.population,
            fitness=self.fitness,
            crossover_probability=.6,
            disturbance_rate=.5,
            number_of_fathers=4,
            number_crossover=100,
            rand=True
        )

        after_cal = calcule_all(res[FIELD_NAMES_CONSTRAINTS], self.transformer)
        res.update(after_cal)
        all = pd.concat([res, self.population], axis=0)
        all.index = range(len(all))
        penalise_res = penalize(all, variations + [(0, 1900), (0, 590)])
        ranks = ranking(penalise_res)

        all.update(penalise_res)
        plot(all, ranks, title="Without penalise")
        plot(penalise_res, ranks, title="all data", annotate=True)
        
        fitness = calcule_fitness(
            penalise_res,
            ranks,
            variations_solutions=[(0, 1900), (0, 590)],
            niche_ray=.1,
            alfa=2,
            const_mult=3
        )
        # import ipdb; ipdb.set_trace()

        sec = selection(
            n_population=n_population,
            population=all,
            fitness=fitness,
            ranks=ranks,
            frac_rank_1=1
        )
        # penalise_res = penalize(sec, variations + [(0, 1900), (0, 590)])
        rank_new = ranking(sec)
        # import ipdb; ipdb.set_trace()
        plot(sec, rank_new, title="After selection", annotate=True)
        # plt.show()

