import click
from unittest import TestLoader, runner
# from app.api import app as api
# from api.app import execute_app
import time

@click.group()
def c():
    ...

@c.command()
def ag():
    print("Starting ...")

@c.command()
def tests():
    loader = TestLoader()
    test = loader.discover("tests/")
    testrunner = runner.TextTestRunner(verbosity=3)
    testrunner.run(test)

c()
    
