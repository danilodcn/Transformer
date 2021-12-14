import itertools as it
from typing import List

import numpy as np
import pandas as pd
from pandas.core.algorithms import rank

from utils.constraints import *
from utils.functions import distance_betwen_points, sum_of_integers
from utils.transformer import Transformer


def init_generation(n_population, variations):
    random_values = np.asarray(
        [np.random.uniform(i, j, size=(n_population)) for i, j in variations]
    ).transpose()
    # import ipdb; ipdb.set_trace()
    return pd.DataFrame(
        random_values,
        columns=FIELD_NAMES_CONSTRAINTS
    )


def calcule_all(df: pd.DataFrame, transformer: Transformer):
    # import ipdb; ipdb.set_trace()
    values = np.asarray(
        [transformer.run(row) for row in df.itertuples(name=None)]
    )

    return pd.DataFrame(
        values,
        columns=FIELD_NAMES_SOLUTIONS,
    )


def penalize(population: pd.DataFrame, variations: List, const_mult=1):
    variations_min, variations_max = np.asarray(variations).transpose()
    number_restrictions = (population < variations_min).sum(axis=1) \
        + (population > variations_max).sum(axis=1)
    loss, mass = variations_max[-2:]
    # import ipdb; ipdb.set_trace()

    return pd.DataFrame(
        np.asarray([
            number_restrictions * loss,
            number_restrictions * mass
        ]).transpose() * const_mult,
        columns=FIELD_NAMES_SOLUTIONS
    ) + population[FIELD_NAMES_SOLUTIONS]


def ranking(population: pd.DataFrame):
    population = population[FIELD_NAMES_SOLUTIONS]
    dominance = [__ranking(row, population) for row in population.iterrows()]
    qtd_dominates = sum(dominance)
    ranks = pd.DataFrame(qtd_dominates, columns=["rank"])
    m = -1
    # import ipdb; ipdb.set_trace()
    for i in np.unique(qtd_dominates):
        ranks[ranks["rank"] == i] = m
        m -= 1


    return ranks * -1


def __ranking(row: pd.Series, population: pd.DataFrame=1):
    _, row = row
    verification = row < population
    dominateds = verification.sum(axis=1)
    # import ipdb; ipdb.set_trace()
    return 1*(dominateds == 2)

def calcule_fitness(
        population: pd.DataFrame,
        ranks: pd.DataFrame,
        variations_solutions,
        niche_ray,
        alfa,
        const_mult=2
    ):

    p1, p2 = variations_solutions[0]
    m1, m2 = variations_solutions[1]
    delta_perdas = p1 - p2
    delta_massas = m1 - m2

    ranks_lst = list(ranks["rank"].drop_duplicates())
    ranks_lst.sort()
    
    fitness = pd.DataFrame(
        np.zeros(len(ranks)),
        columns=["fitness"]
    )

    # ranks.sort()
    all_distance = np.zeros((len(ranks), len(ranks)))

    n_items = len(population)
    for rank in ranks_lst:
        set = population[ranks["rank"]==rank]

        solutions_in_rank = len(set)
        n_current = n_items - solutions_in_rank + 1
        # import ipdb; ipdb.set_trace()
        fraction = sum_of_integers(n_current, n_items)  \
            / solutions_in_rank     \
            * const_mult

        n_items = n_current - 1

        for i in set.index:
            perda_i, massa_i = set["PerdasT"][i], set["Mativa"][i]
        
            # import ipdb; ipdb.set_trace()

            if perda_i > p2 or massa_i > m2:
                distance = distance_betwen_points(
                    perda_i, p2, massa_i, m2, delta_perdas, delta_massas
                ) * const_mult
                distance = distance ** alfa
            else:
                distance = 1

            for j in set.index:
                if i == j:
                    continue
                
                perda_j, massa_j = set["PerdasT"][j], set["Mativa"][j]
                distance_ij = distance_betwen_points(
                    perda_i, perda_j, massa_i, massa_j, delta_perdas, delta_massas
                )
                if distance_ij <= niche_ray:
                    distance += 1 - (distance_ij / niche_ray) ** alfa ** alfa

                all_distance[i, j] = distance_ij
            # finaliza os cálculos e adiciona no DataFrame
            fitness.iloc[i] = fraction / distance
    
    # import ipdb; ipdb.set_trace()
    fitness.apply(lambda u: u ** const_mult)
    fitness = fitness / np.sum(fitness)

    return fitness


def __crossover(f1, f2, f3, crossover_probability, disturbance_rate):
    mask = np.random.rand(f1.count()) < crossover_probability
    f0 = disturbance_rate * (f3 - f2) * 1 * mask

    return f0 + f1
    # import ipdb; ipdb.set_trace()


def crossover(
        population: pd.DataFrame,
        fitness: pd.DataFrame,
        crossover_probability,
        disturbance_rate,
        frac_crossover=1,
    ):

    father = population.sample(frac=.7, weights=fitness['fitness'])
    father: pd.DataFrame = father.sample(frac=1)

    n = 2 + (1 + 8 * frac_crossover * len(fitness)) ** .5
    n = int(n / 2 + .5)
    father_fitness: pd.DataFrame = fitness.loc[father.index]

    father_1: pd.DataFrame = father.sample(n=n, weights=father_fitness["fitness"])\
        .sample(frac=1)
    
    crossover_return = pd.DataFrame(columns=population.columns)

    iterator = zip(
        range(int(frac_crossover * len(population))),
        it.cycle(father.index),
        it.combinations(father_1.index, 2),
    )

    for i, f1, fathers_1 in iterator:
        f2, f3 = fathers_1

        # import ipdb; ipdb.set_trace()

        children = __crossover(
            f1=population.iloc[f1],
            f2=population.iloc[f2],
            f3=population.iloc[f3],
            crossover_probability=crossover_probability,
            disturbance_rate=disturbance_rate
        )
        crossover_return.loc[i] = children

    return crossover_return
    # import ipdb; ipdb.set_trace()

def __get_rand_gene(population: pd.DataFrame, fitness):
    fathers = population.sample(frac=.5, weights=fitness['fitness'])
    fathers = fathers.sample(frac=1)

    while True:
        yield fathers.sample(n=1).index[0]

def __rand_crossover(
        fathers: pd.DataFrame,
        crossover_probability,
        disturbance_rate
        ):
    # n = len(fathers) // 2
    columns_number = fathers.columns.__len__()

    f0 = fathers.iloc[0]
    # import ipdb; ipdb.set_trace()

    for i in range(1, len(fathers), 2):
        mask = np.random.rand(columns_number) < crossover_probability
        f0 += disturbance_rate * (
            fathers.iloc[i] - fathers.iloc[i+1]
        ) * 1 * mask

    return f0

def rand_crossover(
        population: pd.DataFrame,
        fitness: pd.DataFrame,
        crossover_probability,
        disturbance_rate,
        number_crossover,
        number_of_fathers,
        rand=False,
    ):

    fathers = population.sample(frac=.7, weights=fitness['fitness'])
    fathers: pd.DataFrame = fathers.sample(frac=1)
    # import ipdb; ipdb.set_trace()

    iterator_fathers = [range(number_crossover)]
    
    if rand:
        iterator_fathers.append(
            __get_rand_gene(population, fitness)
        )
        if number_of_fathers % 2:
            number_of_fathers -= 1
    
    elif not number_of_fathers % 2:
        number_of_fathers += 1
    
    iterator_fathers += [
        it.cycle(
            fathers.sample(frac=1).index
            ) for _ in range(number_of_fathers)
    ]
    crossover_return = pd.DataFrame(columns=population.columns)

    for i, *fathers_index in zip(*iterator_fathers):
        children = __rand_crossover(
            population.loc[fathers_index],
            crossover_probability,
            disturbance_rate
        )
        crossover_return.loc[i] = children

    # import ipdb; ipdb.set_trace()
    return crossover_return

def selection(
        n_population: int,
        population: pd.DataFrame,
        fitness: pd.DataFrame,
        ranks: pd.DataFrame,
        frac_rank_1: float
    ):
    if frac_rank_1 > 1:
        raise ValueError("frac can't less the 1")

    # import ipdb; ipdb.set_trace()

    rank_1 = population[ranks["rank"] == 1].sample(frac=frac_rank_1)
    if len(rank_1) > n_population:
        fitness_new = fitness.loc[rank_1.index]
        
        res = rank_1.sample(n=n_population, weights=fitness_new["fitness"])

    else:
        in_rank_1 = population.index.isin(rank_1.index)

        fitness_new = fitness.iloc[~in_rank_1]
        population_new = population.iloc[~in_rank_1]
        new = population_new.sample(
            n=n_population - len(rank_1),
            weights=fitness_new["fitness"]
        )

        res = pd.concat([new, rank_1], axis=0)

    res.index = range(len(res))
    return res
