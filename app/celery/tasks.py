from app.celery.app import app

@app.task(bind=True)
def task(self):
    return "ola mundo"