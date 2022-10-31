import os
import pandas as pd

def get_teams_data():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        "teams.csv"
    )
    return pd.read_csv(csv_path)


def get_fixture_data():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        "fixtures.csv"
    )
    return pd.read_csv(csv_path)


def get_fifa_rankings_data():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        "fifa_rankings.csv"
    )
    return pd.read_csv(csv_path)


def get_results_data():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        "match_results_since_2018.csv"
    )
    return pd.read_csv(csv_path,parse_dates=['date'])
