import os
import json
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict


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


def get_fifa_rankings_data(source: str = "game") -> pd.DataFrame:
    """
    Get the FIFA rankings, either from FIFA (the organisation), if source != 'game'
    or from the FIFA video game (with default values for teams not in the game)
    if source == 'game'
    """
    # should we use the videogame rankings over the official FIFA (organisation) ones?
    game_rankings = source == "game"
    current_dir = os.path.dirname(__file__)
    filename = "fifa_game_rankings.csv" if game_rankings else "fifa_rankings.csv"
    csv_path = os.path.join(current_dir, "..", "data", filename)
    df = pd.read_csv(csv_path)
    if game_rankings:
        print("Using FIFA videogame rankings")
        # assign default values to teams not otherwise covered, use the same as Qatar
        default_row = df.loc[df.Team == "Qatar"]
        all_teams = get_confederations_data().Team.unique()
        current_teams = df.Team.unique()
        new_teams = list(set(all_teams) - set(current_teams))
        attacks = len(new_teams) * [default_row.Attack.values[0]]
        midfields = len(new_teams) * [default_row.Midfield.values[0]]
        defences = len(new_teams) * [default_row.Defence.values[0]]
        overalls = len(new_teams) * [default_row.Overall.values[0]]
        new_df = pd.DataFrame(
            {
                "Team": new_teams,
                "Attack": attacks,
                "Midfield": midfields,
                "Defence": defences,
                "Overall": overalls,
            }
        )
        df = pd.concat([df, new_df])
        df = df.reset_index(drop=True)
    else:
        print("Using FIFA organisation rankings")
    return df


def get_results_data(
    start_date: str = "2018-06-01",
    end_date: str = "2022-11-20",
    competitions: List[str] = ["W", "C1", "WQ", "CQ", "C2", "F"],
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
        (results_df.date > start_date) & (results_df.date < end_date)
    ]
    # replace any names that we have written differently elsewhere
    results_df.replace("United States", "USA", inplace=True)
    # filter matches with non-fifa recognised teams
    rankings_df = get_fifa_rankings_data("org")
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
