import json
with open("./app/data.json") as file:
	data = json.load(file)

n_generations = data["n_generations"]
n_population = data["n_population"]
n_mutation = data["n_mutation"]

constraints = data["constraints"]

variations = data["variations"]
variations_solutions = data["variations_solutions"]

tables = data["tables"]