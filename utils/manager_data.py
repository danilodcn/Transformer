import json

with open("./app/data.json") as file:
    data = json.load(file)

def manager_data(data: dict = data):
    