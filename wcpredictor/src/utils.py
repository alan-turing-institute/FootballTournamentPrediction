"""
Assorted functions to get the BPL model, and predict results.
"""
import math
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from bpl import NeutralDixonColesMatchPredictor, NeutralDixonColesMatchPredictorWC
from bpl.base import BaseMatchPredictor

from .bpl_interface import WCPred
from .data_loader import (
    get_confederations_data,
    get_fifa_rankings_data,
    get_results_data,
)


def get_and_train_model(
    start_date: str = "2018-06-01",
    end_date: str = "2022-11-20",
    competitions: List[str] = ["W", "C1", "WQ", "CQ", "C2", "F"],
    rankings_source: str = "org",
    epsilon: float = 0.0,
    world_cup_weight: float = 1.0,
    model: BaseMatchPredictor = NeutralDixonColesMatchPredictorWC(),
    **fit_args,
) -> WCPred:
    """
    Use 'competitions' argument to specify which rows to include in training data.
    Key for competitions:
    "W": world cup finals,
    "C1": top-level continental cup,
    "WQ": world cup qualifiers",
    "CQ": continental cup qualifiers"
    "C2": 2nd-tier continental, e.g. UEFA Nations League,
    "F": friendly/other.

    rankings_source determines whether we use the FIFA video game rankings for prior
    values for the covariates ("game"), or use the FIFA organisation ones ("org"), or
    neither (None).
    """
    print("in get_and_train_model")
    results, weights_dict = get_results_data(
        start_date=start_date,
        end_date=end_date,
        competitions=competitions,
        rankings_source=rankings_source,
        world_cup_weight=world_cup_weight,
    )

    print(f"Using {len(results)} rows in training data")
    ratings = get_fifa_rankings_data(rankings_source) if rankings_source else None
    wc_pred = WCPred(
        results=results,
        ratings=ratings,
        epsilon=epsilon,
        world_cup_weight=world_cup_weight,
        weights_dict=weights_dict,
        model=model,
    )
    wc_pred.set_training_data()
    wc_pred.fit_model(**fit_args)

    return wc_pred


def test_model(
    model: BaseMatchPredictor,
    start_date: str = "2018-06-01",
    end_date: str = "2022-11-20",
    competitions: List[str] = ["W", "C1", "WQ", "CQ", "C2", "F"],
    epsilon=0,
    world_cup_weight=1,
    train_end_date=None,
) -> float:
    """
    Compute the log likelihood of real match scores for a model (to use like a loss
    function)

    Use 'competitions' argument to specify which rows to include in training data.
    Key for competitions:
    "W": world cup finals,
    "C1": top-level continental cup,
    "WQ": world cup qualifiers",
    "CQ": continental cup qualifiers"
    "C2": 2nd-tier continental, e.g. UEFA Nations League,
    "F": friendly/other.
    """
    results, _ = get_results_data(
        start_date, end_date, competitions=competitions, rankings_source=None
    )
    results = results[  # only include matches where model knows about both teams
        (results["home_team"].isin(model.teams))
        & (results["away_team"].isin(model.teams))
    ]
    confed = get_confederations_data()
    confed_dict = dict(zip(confed["Team"], confed["Confederation"]))
    test_data = {
        "home_team": np.array(results.home_team),
        "away_team": np.array(results.away_team),
        "home_conf": np.array([confed_dict[team] for team in results.home_team]),
        "away_conf": np.array([confed_dict[team] for team in results.away_team]),
        "home_goals": np.array(results.home_score),
        "away_goals": np.array(results.away_score),
        "neutral": np.array(results.neutral),
        "game_weight": np.array(results.game_weight),
    }

    if isinstance(model, NeutralDixonColesMatchPredictorWC):
        proba = model.predict_score_proba(
            test_data["home_team"],
            test_data["away_team"],
            test_data["home_conf"],
            test_data["away_conf"],
            test_data["home_goals"],
            test_data["away_goals"],
            test_data["neutral"],
        )
    elif isinstance(model, NeutralDixonColesMatchPredictor):
        proba = model.predict_score_proba(
            test_data["home_team"],
            test_data["away_team"],
            test_data["home_goals"],
            test_data["away_goals"],
            test_data["neutral"],
        )
    else:
        proba = model.predict_score_proba(
            test_data["home_team"],
            test_data["away_team"],
            test_data["home_goals"],
            test_data["away_goals"],
        )

    if epsilon != 0 or world_cup_weight != 1:
        # obtain time difference to last date model was trained on, or test start date
        # if not given
        if train_end_date is None:
            ref_date = pd.Timestamp(start_date)
        else:
            ref_date = pd.Timestamp(train_end_date)
        time_diff = (results.date - ref_date) / pd.Timedelta(days=365)
        weight = test_data["game_weight"] * np.exp(-epsilon * time_diff)
    else:
        weight = np.ones(len(proba))

    return (weight * np.log(proba)).sum() / weight.sum()  # weighted mean log likelihood


def forecast_evaluation(
    model: BaseMatchPredictor,
    start_date: str = "2018-06-01",
    end_date: str = "2022-11-20",
    competitions: List[str] = ["W", "C1", "WQ", "CQ", "C2", "F"],
    method: str = "rps",
) -> List[float]:
    """
    Compute the Brier score, or Rank Probability score (RPS) to evaluate the model against
    real match scores for a model (to use like a loss function). By default computes the RPS

    Use 'competitions' argument to specify which rows to include in training data.
    Key for competitions:
    "W": world cup finals,
    "C1": top-level continental cup,
    "WQ": world cup qualifiers",
    "CQ": continental cup qualifiers"
    "C2": 2nd-tier continental, e.g. UEFA Nations League,
    "F": friendly/other.
    """
    if method not in ["rps", "brier"]:
        raise ValueError("method must be either 'brier' or 'rps'")
    results, _ = get_results_data(
        start_date, end_date, competitions=competitions, rankings_source=None
    )
    results = results[  # only include matches where model knows about both teams
        (results["home_team"].isin(model.teams))
        & (results["away_team"].isin(model.teams))
    ]
    confed = get_confederations_data()
    confed_dict = dict(zip(confed["Team"], confed["Confederation"]))
    test_data = {
        "home_team": np.array(results.home_team),
        "away_team": np.array(results.away_team),
        "home_conf": np.array([confed_dict[team] for team in results.home_team]),
        "away_conf": np.array([confed_dict[team] for team in results.away_team]),
        "neutral": np.array(results.neutral),
    }
    # obtain match outcome probabilities from the model
    if isinstance(model, NeutralDixonColesMatchPredictorWC):
        proba = model.predict_outcome_proba(
            test_data["home_team"],
            test_data["away_team"],
            test_data["home_conf"],
            test_data["away_conf"],
            test_data["neutral"],
        )
    elif isinstance(model, NeutralDixonColesMatchPredictor):
        proba = model.predict_outcome_proba(
            test_data["home_team"],
            test_data["away_team"],
            test_data["neutral"],
        )
    else:
        proba = model.predict_outcome_proba(
            test_data["home_team"],
            test_data["away_team"],
        )
    # obtain len(results) x 3 array where each row is the outcome probabilities for each game
    outcome_probs = jnp.concatenate(list(proba.values())).reshape([3, len(results)])
    outcome_probs = outcome_probs.transpose()
    # obtain actual match outcomes from the test data
    outcome = [
        jnp.array([1, 0, 0])
        if game["home_score"] > game["away_score"]
        else jnp.array([0, 1, 0])
        if game["home_score"] == game["away_score"]
        else jnp.array([0, 0, 1])
        for index, game in results.iterrows()
    ]

    metrics = []
    for i in range(len(results)):
        # fix any nans (happens when have two very lobsided teams - computational underflow)
        prediction = outcome_probs[i, :]
        if math.isnan(prediction[0].item()):
            prediction = prediction.at[0].set(1 - (prediction[1] + prediction[2]))
        elif math.isnan(prediction[1].item()):
            prediction = prediction.at[1].set(1 - (prediction[0] + prediction[2]))
        elif math.isnan(prediction[2].item()):
            prediction = prediction.at[2].set(1 - (prediction[0] + prediction[1]))
        # compute metric
        if method == "brier":
            metrics.append(((prediction - outcome[i]) ** 2).sum().item())
        elif method == "rps":
            metrics.append(
                ((prediction.cumsum() - outcome[i].cumsum()) ** 2)[:2].sum().item() / 2
            )

    return metrics


def find_group(team, teams_df):
    """
    Look in teams_df and find the group for a given team

    Parameters
    ==========
    team: str, team name, as given in teams.csv
    teams_df: Pandas dataframe

    Returns
    =======
    group_name: str, "A"-"H", or None if team not found
    """
    for idx, row in teams_df.iterrows():
        if row.Team == team:
            return row.Group
    print("Unable to find {} in teams.csv".format(team))
    return None


def sort_teams_by(table_dict, metric):
    """
    Given a dictionary in the same format as self.table (i.e. keyed by
    team name), return a list of dicts
    [{"team": <team_name>, "points": <points>, ...},{...}]
    in order of whatever metric is supplied

    Parameters
    ==========
    table_dict: dict, keyed by team name, in the same format as self.table

    Returns
    =======
    team_list: list of dicts, [{"team": "abc","points": x},...], ordered
               according to metric
    """
    if metric not in ["points", "goal_difference", "goals_for", "goals_against"]:
        raise RuntimeError(f"unknown metric for sorting: {metric}")
    team_list = [{"team": k, **v} for k, v in table_dict.items()]
    team_list = sorted(team_list, key=lambda t: t[metric], reverse=True)
    return team_list


def predict_knockout_match(
    wc_pred: WCPred, team_1: str, team_2: str, seed: Optional[int] = None
) -> str:
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    winning_team: str, one of team_1 or team_2
    """
    return wc_pred.get_fixture_probabilities(
        fixture_teams=[(team_1, team_2)], knockout=True, seed=seed
    )["simulated_outcome"][0]


def predict_group_match(
    wc_pred: WCPred, team_1: str, team_2: str, seed: Optional[int] = None
) -> Tuple[int, int]:
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    score_1, score_2: both int, score for each team
    """
    return wc_pred.get_fixture_goal_probabilities(
        fixture_teams=[(team_1, team_2)], seed=seed
    )[1][0]


def get_difference_in_stages(stage_1: str, stage_2: str) -> int:
    """
    Give an integer value to the differences between two
    'stages' i.e. how far a team got in the tournament.
    This can be used to calculate a loss function, i.e. difference
    between predicted and actual results for past tournaments.

    Parameters
    ==========
    stage_1, stage_2: both str, can be "G","R16","QF","SF","RU","W"

    Returns
    =======
    diff: int, how far apart the two stages are.
    """
    stages = ["G", "R16", "QF", "SF", "RU", "W"]
    if stage_1 not in stages and stage_2 in stages:
        raise RuntimeError(f"Unknown value for stage - must be in {stages}")
    return abs(stages.index(stage_1) - stages.index(stage_2))
