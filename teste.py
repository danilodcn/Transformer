from unittest import TestCase
from utils.generation import init_generation, calcule_all
from tests.data_for_testes import *
from utils.transformer import Transformer
from utils.constraints import FIELD_NAMES_CONSTRAINTS

population = init_generation(n_population*100, variations)
transformer = Transformer(constraints=constraints, tables=tables)
res = calcule_all(population[FIELD_NAMES_CONSTRAINTS], transformer)

# print(res)