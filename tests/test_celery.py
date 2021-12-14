import json
from unittest import TestCase

from app.celery.generation import start_generation
from app.celery.tasks import task
from matplotlib import pyplot as plt


class TestCelery(TestCase):
    
    def test_initial(self):
        task.delay()

    def test_run_many(self):
        with open("./app/data.json") as file:
            data = json.load(file)
        # data["n_population"] = 30
        import time
        t = time.time()
        start_generation.delay(data)
        t0 = time.time()

        print("demorou", round(t0 -t, 4), "segundos")
        # start_generation.delay(data)
        plt.show()