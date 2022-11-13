import json
import os
from typing import List

import pandas as pd



def get_teams_data(year: str = "2022") -> pd.DataFrame:
    if year not in ["2014", "2018", "2022"]:
        raise RuntimeError("Unknown year " + year)
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "data", f"teams_{year}.csv")
    return pd.read_csv(csv_path)


def get_fixture_data(year: str = "2022") -> pd.DataFrame:
    if year not in ["2014", "2018", "2022"]:
        raise RuntimeError("Unknown year " + year)
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "data", f"fixtures_{year}.csv")
    return pd.read_csv(csv_path)


def get_confederations_data() -> pd.DataFrame:
    """
    Which teams belong to which federations
    """
    current_dir = os.path.dirname(__file__)
    filename = "confederations.csv"
    csv_path = os.path.join(current_dir, "..", "data", filename)
    df = pd.read_csv(csv_path)
    return df


def load_game_rankings() -> pd.DataFrame:
    print("Using FIFA videogame rankings")
    current_dir = os.path.dirname(__file__)
    filename = "fifa_game_rankings.csv"
    csv_path = os.path.join(current_dir, "..", "data", filename)
    df = pd.read_csv(csv_path)
    # assign default values to teams not otherwise covered
    confederations = get_confederations_data()
    confed_dict = dict(zip(confederations.Team, confederations.Confederation))
    all_teams = confederations.Team.unique()
    current_teams = df.Team.unique()
    new_teams = list(set(all_teams) - set(current_teams))
    teams = []
    attacks = []
    midfields = []
    defences = []
    overalls = []
    for conf in set(confederations.Confederation):
        # define default value for Fifa ratings conditional on their confederation
        if conf == "AFC":
            default = 60
        elif conf == "CAF":
            default = 60
        elif conf == "CONCACAF":
            default = 60
        elif conf == "CONMEBOL":
            default = 65
        elif conf == "OFC":
            default = 50
        elif conf == "UEFA":
            default = 65
        new_teams_in_conf = [team for team in new_teams if confed_dict[team] == conf]
        teams += new_teams_in_conf
        attacks += len(new_teams_in_conf) * [default]
        midfields += len(new_teams_in_conf) * [default]
        defences += len(new_teams_in_conf) * [default]
        overalls += len(new_teams_in_conf) * [default]
    new_df = pd.DataFrame(
        {
            "Team": teams,
            "Attack": attacks,
            "Midfield": midfields,
            "Defence": defences,
            "Overall": overalls,
        }
    )
    df = pd.concat([df, new_df]).reset_index(drop=True)
    return df


def load_org_rankings() -> pd.DataFrame:
    print("Using FIFA organisation rankings")
    current_dir = os.path.dirname(__file__)
    filename = "fifa_rankings.csv"
    csv_path = os.path.join(current_dir, "..", "data", filename)
    df = pd.read_csv(csv_path)
    return df


def get_fifa_rankings_data(source: str = "game") -> pd.DataFrame:
    """
    Get the FIFA rankings, either from FIFA (the organisation), if source == 'org'
    or from the FIFA video game (with default values for teams not in the game)
    if source == 'game', or combine both if source == 'both'
    """
    if source == "game":
        return load_game_rankings()
    elif source == "org":
        return load_org_rankings()
    elif source == "both":
        return pd.merge(
            load_game_rankings(), load_org_rankings(), how="inner", on="Team"
        )

def get_results_data(
    start_date: str = "2018-06-01",
    end_date: str = "2022-11-20",
    competitions: List[str] = ["W", "C1", "WQ", "CQ", "C2", "F"],
    rankings_source: str = "org",
) -> pd.DataFrame:
    """
    filter the results dataframe by date and competition.
    Key for competitions:
    "W": world cup finals,
    "C1": top-level continental cup,
    "WQ": world cup qualifiers",
    "CQ": continental cup qualifiers"
    "C2": 2nd-tier continental, e.g. UEFA Nations League,
    "F": friendly/other.
    """
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "data", "results.csv")
    results_df = pd.read_csv(csv_path, parse_dates=["date"])
    # get an index of what competition is in what category
    json_path = os.path.join(current_dir, "..", "data", "competition_index.json")
    competitions_index = json.load(open(json_path))
    # filter by date
    results_df = results_df[
        (results_df.date >= start_date) & (results_df.date <= end_date)
    ]
    # replace any names that we have written differently elsewhere
    results_df.replace("United States", "USA", inplace=True)
    results_df.replace(
        "United States Virgin Islands", "USA Virgin Islands", inplace=True
    )
    # filter matches with non-fifa recognised teams
    if rankings_source:
        rankings_df = get_fifa_rankings_data(rankings_source)
        fifa_teams = rankings_df.Team.values
        results_df = results_df[
            (results_df.home_team.isin(fifa_teams))
            & (results_df.away_team.isin(fifa_teams))
        ]
    # filter by competition
    comp_filter = [competitions_index[comp] for comp in competitions]
    # flatten this nested list
    comp_filter = [comp for complist in comp_filter for comp in complist]
    results_df = results_df[results_df.tournament.isin(comp_filter)]

    results_df = results_df.reset_index(drop=True)
    return results_df


def get_wcresults_data(year: str) -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "data", f"wcresults_{year}.csv")
    wcresults_df = pd.read_csv(csv_path)
    return wcresults_df
