#!/usr/bin/env python

"""
API for calling airsenal functions.
HTTP requests to the endpoints defined here will give rise
to calls to functions in api_utils.py
"""
import json
from uuid import uuid4

from flask import Blueprint, Flask, jsonify, request, session
from flask_cors import CORS
from flask_session import Session

from wcpredictor import get_teams_data, get_fixture_data

blueprint = Blueprint("wcpred", __name__)

def create_response(orig_response):
    """
    Add headers to the response
    """
    response = jsonify(orig_response)
    response.headers.add(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, x-auth",
    )
    return response


@blueprint.route("/teams", methods=["GET"])
def get_team_list():
    """
    Return a list of all teams
    """
    teams_df = get_teams_data()
    team_list = list(teams_df.Team.values)
    return create_response(team_list)


@blueprint.route("/teams/<group>", methods=["GET"])
def get_team_list_for_group(group):
    """
    Return a list of all teams in specified group
    """
    teams_df = get_teams_data()
    group_df = teams_df[teams_df.Group==group]
    team_list = list(group_df.Team.values)
    return create_response(team_list)


@blueprint.route("/fixtures", methods=["GET"])
def get_fixtures():
    """
    Return a list of all fixtures
    """
    fixtures_df = get_fixture_data()
    fixture_list = []
    for _, row in fixtures_df.iterrows():
        fixture_list.append({"team_1": row.Team_1,
                             "team_2": row.Team_2,
                             "date": row.Date
                             })
    return create_response(fixture_list)


@blueprint.route("/fixtures/<group>", methods=["GET"])
def get_fixtures_for_group(group):
    """
    Return a list of all fixtures in specified group
    """
    teams_df = get_teams_data()
    fixtures_df = get_fixture_data()
    group_df = teams_df[teams_df.Group==group]
    team_list = list(group_df.Team.values)
    group_fixtures_df = fixtures_df[fixtures_df.Team_1.isin(team_list)]
    fixture_list = []
    for _, row in group_fixtures_df.iterrows():
        fixture_list.append({"team_1": row.Team_1,
                             "team_2": row.Team_2,
                             "date": row.Date
                             })
    return create_response(fixture_list)


def create_app(name=__name__):
    app = Flask(name)
    app.config["SESSION_TYPE"] = "filesystem"
    app.secret_key = "blah"
    CORS(app, supports_credentials=True)
    app.register_blueprint(blueprint)
    Session(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
