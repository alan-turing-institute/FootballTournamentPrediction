import os
import json
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict

def get_teams_data(year: str = "2022") -> pd.DataFrame:
    if year not in ["2014","2018","2022"]:
        raise RuntimeError("Unknown year "+year)
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        f"teams_{year}.csv"
    )
    return pd.read_csv(csv_path)


def get_fixture_data(year: str = "2022") -> pd.DataFrame:
    if year not in ["2014","2018","2022"]:
        raise RuntimeError("Unknown year "+year)
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        f"fixtures_{year}.csv"
    )
    return pd.read_csv(csv_path)


def get_fifa_rankings_data() -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        "fifa_rankings.csv"
    )
    return pd.read_csv(csv_path)


def get_confederations_data() -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        "confederations.csv"
    )
    return pd.read_csv(csv_path)

def get_results_data(
        start_date: str = "2018-06-01",
        end_date: str = "2022-11-20",
        competitions:List[str] = ["W","C1","WQ","CQ","C2","F"]
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
    csv_path = os.path.join(
        current_dir, "..","data",
        "results.csv"
    )
    results_df = pd.read_csv(csv_path,parse_dates=['date'])
    # get an index of what competition is in what category
    json_path = os.path.join(
        current_dir, "..","data",
        "competition_index.json"
    )
    competitions_index = json.load(open(json_path))
    # filter by date
    results_df = results_df[(results_df.date > start_date) &
                            (results_df.date < end_date)]
    # replace any names that we have written differently elsewhere
    results_df.replace("United States", "USA", inplace=True)
    # filter matches with non-fifa recognised teams
    rankings_df = get_fifa_rankings_data()
    fifa_teams = (rankings_df.Team.values)
    results_df = results_df[(results_df.home_team.isin(fifa_teams)) &
                            (results_df.away_team.isin(fifa_teams))]
    # filter by competition
    comp_filter = [competitions_index[comp] for comp in competitions]
    # flatten this nested list
    comp_filter = [comp for complist in comp_filter for comp in complist]
    results_df = results_df[results_df.tournament.isin(comp_filter)]

    results_df.reset_index(drop=True)
    return results_df


def get_wcresults_data(year: str) -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        f"wcresults_{year}.csv"
    )
    wcresults_df = pd.read_csv(csv_path)
    return wcresults_df
