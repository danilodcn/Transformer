from os import environ
from dotenv import load_dotenv

load_dotenv()

# print("Aqui")
class Config:
    FLASK_APP = environ.get("FLASK_APP")
    FLASK_ENV = environ.get("FLASK_ENV")
    FLASK_SECRET_KEY = environ.get("FLASK_SECRET_KEY")

    STATIC_FOLDER = "static"
    TEMPLATES_FOLDER = "templates"
