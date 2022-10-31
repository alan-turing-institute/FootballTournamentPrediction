"""
Assorted functions to get the BPL model, and predict results.
"""
from typing import Optional, Union, List, Tuple

from .data_loader import (
    get_results_data,
    get_teams_data,
    get_fifa_rankings_data
)
from .bpl_interface import WCPred


def get_and_train_model(only_wc_teams: bool = True,
                        use_ratings: bool = True):
    results = get_results_data()
    teams = list(get_teams_data().Team)
    ratings = get_fifa_rankings_data()
    if (only_wc_teams and use_ratings):
        wc_pred = WCPred(results=results,
                        teams=teams,
                        ratings=ratings)
    elif only_wc_teams and not use_ratings:
        wc_pred = WCPred(results=results,
                        teams=teams)
    else:
        wc_pred = WCPred(results=results)
    wc_pred.set_training_data()
    wc_pred.fit_model()
    return wc_pred


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
    if not metric in ["points","goal_difference","goals_for","goals_against"]:
        raise RuntimeError(f"unknown metric for sorting: {metric}")
    team_list = [{"team": k, **v} for k,v in table_dict.items()]
    team_list = sorted(team_list, key=lambda t: t[metric],reverse=True)
    return team_list


def predict_knockout_match(wc_pred: WCPred,
                           team_1: str,
                           team_2: str,
                           seed: Optional[int] = None) -> str:
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    winning_team: str, one of team_1 or team_2
    """
    return wc_pred.get_fixture_probabilities(
        fixture_teams = [(team_1, team_2)],
        knockout = True,
        seed = seed
    )["simulated_outcome"][0]


def predict_group_match(wc_pred: WCPred,
                        team_1: str,
                        team_2: str,
                        seed: Optional[int] = None) -> Tuple[int, int]:
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    score_1, score_2: both int, score for each team
    """
    return wc_pred.get_fixture_goal_probabilities(
        fixture_teams = [(team_1, team_2)],
        seed = seed
    )[1][0]
