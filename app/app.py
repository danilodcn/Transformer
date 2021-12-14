from flask import Flask
from app.blueprints.routes import routes_bp
from app.app_dash.dashboard import init_dash

def init_app() -> Flask:
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("app_config.Config")

    app.register_blueprint(routes_bp)

    init_dash(app)
    return app