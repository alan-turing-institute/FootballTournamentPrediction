#!/usr/bin/env python
import os
from re import U
import pandas as pd
import argparse

from wcpredictor  import get_teams_data
from wcpredictor  import Tournament
from wcpredictor  import get_and_train_model


def main():
    parser = argparse.ArgumentParser("Simulate multiple World Cups")
    parser.add_argument("--num_simulations",
                        help="How many simulations to run",
                        type=int)
    parser.add_argument("--tournament_year",
                        help="Which world cup to simulate? 2018 or 2022",
                        choices={"2018","2022"},
                        default="2022")
    parser.add_argument("--training_data_start",
                        help="earliest date for training data",
                        default="2018-06-30")
    parser.add_argument("--training_data_end",
                        help="latest date for training data",
                        default="2022-11-20")
    parser.add_argument("--output_csv",
                        help="Path to output CSV file",
                        default="sim_results.csv")
    parser.add_argument("--only_wc_teams",
                        help="If set, only fits model to teams playing in the World Cup",
                        action="store_true")
    parser.add_argument("--use_ratings",
                        help="If set, model is fitted using the Fifa rankings of each team",
                        action="store_true")
    args = parser.parse_args()
    model = get_and_train_model(only_wc_teams = args.only_wc_teams,
                                use_ratings = args.use_ratings,
                                start_date = args.training_data_start,
                                end_date = args.training_data_end)
    teams_df = get_teams_data(args.tournament_year)
    teams = list(teams_df.Team.values)
    team_results = {
        team: {
            "G": 0,
            "R16":0,
            "QF": 0,
            "SF": 0,
            "RU": 0,
            "W": 0
        } for team in teams
    }
    for _ in range(args.num_simulations):
        t = Tournament(args.tournament_year)
        t.play_group_stage(model)
        t.play_knockout_stages(model)
        for team in teams:
            result = t.get_furthest_position_for_team(team)
            team_results[team][result] += 1
    team_records = [{"team": k, **v} for k,v in team_results.items()]
    df = pd.DataFrame(team_records)
    df.to_csv(args.output_csv)


if __name__ == "__main__":
    main()
