from flask import Blueprint, render_template


routes_bp = Blueprint("routes", __name__, url_prefix="/")


@routes_bp.get("/")
def home_routes():
    return render_template("home.html", title="Transformer")

@routes_bp.get("/site-police/privacy")
def privacy():
    return render_template("privacy.html")
