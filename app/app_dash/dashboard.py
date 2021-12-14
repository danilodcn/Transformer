from dash import Dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def init_dash(server):
    app = Dash(
        server=server,
        routes_pathname_prefix="/dashboard/",
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    app.layout = dbc.Container(
        dbc.Alert("Ola mundo", color="success"),
        className="p-5"
    )

    