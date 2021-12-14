import json
from typing import Any, Dict
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from utils.constraints import FIELD_NAMES_CONSTRAINTS
from utils.generation import calcule_all, init_generation, penalize, rand_crossover, ranking, calcule_fitness, selection

from app.celery.app import app
from celery.contrib import rdb
from utils.plot import plot

from utils.transformer import Transformer


@app.task(bind=True)
def start_generation(self, data: Dict[str, Any]):
    print("starting ...")
    try:
        constraints = data["constraints"]
        
        n_generations = data["n_generations"]
        n_mutation = data["n_mutation"]
        n_population = data["n_population"]

        tables = data["tables"]
        user_id = data["user_id"]

        variations = data["variations"]
        variations_solutions = data["variations_solutions"]
        
        population = init_generation(
            n_population=n_population,
            variations=variations,
        )
        # import ipdb; ipdb.set_trace()

        # variáveis necessárias
        niche_ray = .02         # distancia máxima entre os genes
        alfa = 2                # constante usada no fitness
        const_mult=3            # favorece os indivíduos de ranks superiores
        frac_rank_1=1          # percentagem de indivíduos que do 
                                # rank 1 que serão salvos a cada geração

        # Variaveis do crossover
        crossover_probability=0.35
        disturbance_rate=1.5
        number_crossover=data["number_crossover"]
        number_of_fathers=5

    except Exception as error:
        ...
        raise ValueError("Error" + str(error))
        # return {"status": "error", "mesage": str(error)}
    for current_generation in range(n_generations):

        population = run_generation(
                user_id=user_id,
                n_population=n_population,
                n_generations=n_generations,
                constraints=constraints,
                current_generation=current_generation,
                variations=variations,
                variations_solutions=variations_solutions,
                tables=tables,
                n_mutation=n_mutation,
                # current_population=population.to_json(),
                population=population,
                niche_ray=niche_ray,
                alfa=alfa,
                const_mult=const_mult,
                crossover_probability=crossover_probability,
                disturbance_rate=disturbance_rate,
                number_crossover=number_crossover,
                number_of_fathers=number_of_fathers,
                frac_rank_1=frac_rank_1,
            )
        
        data = {
            "msg": "ok",
            "current_generation": current_generation + 1,
            "population": population.to_json(),
        }
        notify.delay(data)
        plt.show()
    return {"status": "ok", "mesage": "created"}

    # except Exception as error:
    #     return {"status": "error", "mesage": str(error)}
    #     raise ValueError("Error " + str(error))


@app.task()
def run_generation(
        user_id: int,
        n_population: int,
        n_generations: int,
        constraints: list,
        current_generation: int,
        variations: list,
        variations_solutions: list,
        tables: dict,
        n_mutation: float,
        population: pd.DataFrame,
        niche_ray: float,
        alfa: float,
        const_mult: float,
        crossover_probability: float,
        disturbance_rate: float,
        number_crossover: int,
        number_of_fathers: int,
        frac_rank_1: float,
    ) -> pd.DataFrame:

    current_generation += 1
    print(f"starting generation {current_generation}...")

    # import ipdb; ipdb.set_trace()

    # population = pd.DataFrame(columns=FIELD_NAMES_CONSTRAINTS, dtype=np.float64)\
    #     .from_dict(json.loads(current_population))
    transformer = Transformer(constraints=constraints, tables=tables)

    population.reset_index(drop=True, inplace=True)
    # calculo da massa ativa e perdas internas
    res_calcule = calcule_all(
        population[FIELD_NAMES_CONSTRAINTS],
        transformer
    )

    res_calcule.reset_index(drop=True, inplace=True)

    population["PerdasT"] = res_calcule["PerdasT"]
    population["Mativa"] = res_calcule["Mativa"]
    # population = pd.concat(
    #     [population, res_calcule],
    #     axis=1,
    # )

    # penalizar
    # import ipdb; ipdb.set_trace()

    res_penalize = penalize(
        population=population,
        variations=variations + variations_solutions,
        const_mult=const_mult
    )

    res_ranking = ranking(population)

    # plot(population, res_ranking, title="After first ranking", annotate=True)

    res_fitness = calcule_fitness(
        population=res_penalize,
        ranks=res_ranking,
        variations_solutions=variations_solutions,
        niche_ray=niche_ray,
        alfa=alfa,
        const_mult=const_mult
    )
    # import ipdb; ipdb.set_trace()

    # fazer o crossover
    res_crossover = rand_crossover(
        population=population,
        fitness=res_fitness,
        crossover_probability=crossover_probability,
        disturbance_rate=disturbance_rate,
        number_crossover=number_crossover,
        number_of_fathers=number_of_fathers,
        rand=False,
    )
    
    res_crossover_after_cal = calcule_all(
        res_crossover[FIELD_NAMES_CONSTRAINTS],
        transformer
    )
    res_crossover.update(res_crossover_after_cal)

    # juntar os resultados do crossover com a população

    population = pd.concat(
        [population, res_crossover],
        axis=0
    )
    
    population.index = range(len(population))
    
    res_penalize_after_crossover = penalize(
        population=population,
        variations=variations + variations_solutions,
        const_mult=const_mult
    )
    
    res_ranking_after_crossover = ranking(res_penalize_after_crossover)
    
    res_fitness_after_crossover = calcule_fitness(
        population=res_penalize_after_crossover,
        ranks=res_ranking_after_crossover,
        variations_solutions=variations_solutions,
        niche_ray=niche_ray,
        alfa=alfa,
        const_mult=const_mult,
    )

    # plot(res_penalize_after_crossover, res_ranking_after_crossover, annotate=True)
    # plot(population, res_ranking_after_crossover, annotate=True)
    # plt.show()

    res_selection = selection(
        n_population=n_population,
        population=population,
        fitness=res_fitness_after_crossover,
        ranks=res_ranking_after_crossover,
        frac_rank_1=frac_rank_1
    )
    # import ipdb; ipdb.set_trace()

    res_ranking = ranking(res_selection)

    plot(res_selection, res_ranking, title=f"After generation {current_generation}", annotate=True)

    # import ipdb; ipdb.set_trace()
    # rdb.set_trace()
    print(f"End generation {current_generation}")
    
    return res_selection

    notify.delay(f"End generation {current_generation}")

    if current_generation >= n_generations:
        # plot(res_selection, res_ranking, title=f"After generation {current_generation}", annotate=True)
        return {
            "msg": "Terminated",
            "n_population": n_population,
            "max_generation": n_generations,
        }


    # run_generation.delay(
    #         user_id=user_id,
    #         n_population=n_population,
    #         n_generations=n_generations,
    #         constraints=constraints,
    #         current_generation=current_generation,
    #         variations=variations,
    #         variations_solutions=variations_solutions,
    #         tables=tables,
    #         n_mutation=n_mutation,
    #         current_population=res_selection.to_json(),
    #         niche_ray=niche_ray,
    #         alfa=alfa,
    #         const_mult=const_mult,
    #         crossover_probability=crossover_probability,
    #         disturbance_rate=disturbance_rate,
    #         number_crossover=number_crossover,
    #         number_of_fathers=number_of_fathers,
    #         frac_rank_1=frac_rank_1,
    #     )

        # run_generation.delay(n_population+1, max_generation)

        # Enviar relatório e salvar no banco de Dados
    return "passei para a próxima ..." 

@app.task(bind=True)
def notify(self, data: dict):
    return data