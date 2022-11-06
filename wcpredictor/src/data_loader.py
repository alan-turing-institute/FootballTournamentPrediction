import os
import pandas as pd

def get_teams_data(year: str = "2022") -> pd.DataFrame:
    if year not in ["2014","2018","2022"]:
        raise RunTimeError("Unknown year "+year)
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        f"teams_{year}.csv"
    )
    return pd.read_csv(csv_path)


def get_fixture_data(year: str = "2022") -> pd.DataFrame:
    if year not in ["2014","2018","2022"]:
        raise RunTimeError("Unknown year "+year)
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


def get_results_data(
        start_date: str = "2018-06-01",
        end_date: str = "2022-11-20") -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir, "..","data",
        "results.csv"
    )
    results_df = pd.read_csv(csv_path,parse_dates=['date'])
    # filter by date
    results_df = results_df[(results_df.date > start_date) &
                            (results_df.date < end_date)]
    # filter matches with non-fifa recognised teams
    rankings_df = get_fifa_rankings_data()
    fifa_teams = (rankings_df.Team.values)
    results_df = results_df[(results_df.home_team.isin(fifa_teams)) &
                            (results_df.away_team.isin(fifa_teams))]
    results_df.reset_index(drop=True)
    return results_df
