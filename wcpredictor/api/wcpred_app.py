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

from wcpredictor import (
    get_teams_data,
    get_fixture_data,
    get_and_train_model,
    predict_group_match,
    predict_knockout_match,
    predict_score_probabilities,
    Tournament
)

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


@blueprint.route("/tournament", methods=["GET"])
def create_tournament():
    """
    set the tournament
    """
    session["tournament"] = Tournament()
    return create_response("created new tournament")


@blueprint.route("/train", methods=["GET"])
def get_model():
    """
    get and train the model
    """
    session["model"] = get_and_train_model()
    return create_response("model trained ok")


@blueprint.route("/score_probs/<team_1>/<team_2>", methods=["GET"])
def get_score_probabilities(team_1, team_2):
    """
    get score probabilities for a match
    """
    if not "model" in session.keys():
        get_model()
    probs = session["model"].get_fixture_goal_probabilities(
        [(team_1, team_2)])[0][0]
    # convert numpy types to regular python ints and floats
    output = {}
    for k,v in probs.items():
        output[k] = {}
        for kk,vv in v.items():
            output[k][int(kk)] = float(vv)
    return create_response(output)


@blueprint.route("/predict_group_match/<team_1>/<team_2>", methods=["GET"])
def get_match_scores(team_1, team_2):
    """
    get scores for a group match
    """
    if not "model" in session.keys():
        get_model()
    scores = predict_group_match(session["model"], team_1, team_2)
    scores = [int(score) for score in scores]
    return create_response(scores)


@blueprint.route("/predict_knockout_match/<team_1>/<team_2>", methods=["GET"])
def get_match_result(team_1, team_2):
    """
    get winner for a knockout match
    """
    if not "model" in session.keys():
        get_model()
    winner = predict_knockout_match(session["model"], team_1, team_2)
    return create_response(winner)


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


def get_ordered_table(group, tournament):
    """
    combine 'standings' and 'table' to get an ordered table
    """
    tournament.groups[group].calc_standings()
    standings = tournament.groups[group].standings
    table = tournament.groups[group].table
    print("TABLE", table)
    ordered_table = [
        {
            "position": k,
            "team": v,
            "points": table[v]["points"],
            "gs": table[v]["goals_for"],
            "ga": table[v]["goals_against"]
        } for k,v in standings.items()
    ]
    return ordered_table

@blueprint.route("/group/<group>", methods=["GET"])
def get_group_table(group):
    """
    Return the standings for a group
    """
    if not "tournament" in session.keys():
        create_tournament()
    ordered_table = get_ordered_table(group, session["tournament"])
    return create_response(ordered_table)


@blueprint.route("/groups", methods=["GET"])
def get_all_group_tables():
    """
    Return the standings for all groups
    """
    if not "tournament" in session.keys():
        create_tournament()
    tables = {}
    for group in ["A","B","C","D","E","F","G","H"]:
        tables[group] = get_ordered_table(group, session["tournament"])
    return create_response(tables)


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

@blueprint.route("/matches/next", methods=["GET", "POST"])
def next_match():
    """
    Find the next fixture to be played.
    """
    if not "tournament" in session.keys():
        create_tournament()
    if request.method == "GET":
        match = session["tournament"].get_next_match()
        return create_response(match)
    else:
        if not "model" in session.keys():
            get_model()
        result = session["tournament"].play_next_match(session["model"])
        return create_response(result)



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
