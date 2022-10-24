#!/usr/bin/env python
import os
import pandas as pd
import argparse

from tournament import Tournament, get_teams_data, get_and_train_model


def main(args):
    model = get_and_train_model()
    teams_df = get_teams_data()
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
        t = Tournament()
        t.play_group_stage(model)
        t.play_knockout_stages(model)
        for team in teams:
            result = t.get_furthest_position_for_team(team)
            team_results[team][result] += 1
    team_records = [{"team": k, **v} for k,v in team_results.items()]
    df = pd.DataFrame(team_records)
    df.to_csv(args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulate multiple World Cups")
    parser.add_argument("--num_simulations",help="How many simulations to run", type=int)
    parser.add_argument("--output_csv", help="Path to output CSV file", default="sim_results.csv")
    args = parser.parse_args()
    main(args)
