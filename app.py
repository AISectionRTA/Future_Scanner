import dash
import dash_bootstrap_components as dbc
import flask

server = flask.Flask(__name__, static_folder='assets')

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.FLATLY],

)
