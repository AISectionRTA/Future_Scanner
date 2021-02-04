import json

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask import Flask, request, jsonify
from flask import session
# from flask.ext.session import Session
from flask_cors import CORS
import pandas as pd

from app import app, server
from apps import app2

CORS(server)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

login_creds = pd.read_csv('login_creds.csv', sep="|")


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        # Bus Route Optimization
        return app2.layout

    else:
        return '404'


def validate_creds(user_name, password):
    if user_name not in list(login_creds.username):
        return {"message": "invalid"}
    else:
        if password == login_creds[login_creds.username == user_name].password.iloc[0]:
            return {"message": "success", "usertype": login_creds[(login_creds.username == user_name) & (
                    login_creds.password == password)].usertype.iloc[0],
                    "name": login_creds[(login_creds.username == user_name) & (
                            login_creds.password == password)].name.iloc[0], "username": user_name,
                    "pages_accessible": login_creds[
                        (login_creds.username == user_name) & (login_creds.password == password)].pages_accessible.iloc[
                        0]}
        else:
            return {"message": "invalid"}


@server.route('/login', methods=['GET'])
def login_user():
    user_name = request.args.get('username')
    password = request.args.get('password')
    resp = validate_creds(user_name, password)

    return jsonify(resp)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5000)
