import os

def detect_tasks(path):
    tasks = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename != "__init__.py" and filename.endswith(".py"):
                task = os.path.join(root, filename) \
                    .replace(os.getcwd() + "/", "") \
                    .replace("/", ".")  \
                    .replace(".py", "")
                tasks.append(task)
    
    return tuple(tasks)
## Broker settings.
broker_url = 'amqp://guest:guest@localhost:5672//'

# List of modules to import when celery starts.
imports = detect_tasks("app/celery")
print(imports)

## Using the database to store task state and results.
result_backend = 'db+sqlite:///results.db'