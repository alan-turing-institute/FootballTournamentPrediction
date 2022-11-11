#!/usr/bin/env python
import os
from re import U
import pandas as pd
import argparse

from wcpredictor  import (
    get_teams_data,
    get_wcresults_data,
    Tournament,
    get_and_train_model,
    get_difference_in_stages
)


def get_cmd_line_args():
    parser = argparse.ArgumentParser("Simulate multiple World Cups")
    parser.add_argument("--num_simulations",
                        help="How many simulations to run",
                        type=int)
    parser.add_argument("--tournament_year",
                        help="Which world cup to simulate? 2014, 2018 or 2022",
                        choices={"2014","2018","2022"},
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
    parser.add_argument("--output_loss_txt",
                        help="Path to output txt file of loss function vals",
                        default="sim_results_loss.txt")
    parser.add_argument("--only_wc_teams",
                        help="If set, only fits model to teams playing in the World Cup",
                        action="store_true")
    parser.add_argument("--dont_use_ratings",
                        help="If set, model is fitted without using the Fifa rankings of each team",
                        action="store_true")
    parser.add_argument("--include_competitions",
                        help="comma-separated list of competitions to include in training data",
                        default="W,C1,WQ,CQ,C2,F")
    parser.add_argument("--exclude_competitions",
                        help="comma-separated list of competitions to exclude from training data")

    args = parser.parse_args()
    return args

def main():
    args = get_cmd_line_args()
    # use the fifa ratings as priors?
    use_ratings = False if args.dont_use_ratings else True
    # list of competitions to include
    comps = args.include_competitions.split(",")
    if args.exclude_competitions:
        exclude_comps = args.exclude_competitions.split(",")
        for comp in exclude_comps:
            comps.remove(comp)
    model = get_and_train_model(only_wc_teams = args.only_wc_teams,
                                use_ratings = use_ratings,
                                start_date = args.training_data_start,
                                end_date = args.training_data_end,
                                competitions = comps)
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
    wcresults_df = None
    loss_values = []
    if args.tournament_year != "2022":
        wcresults_df = get_wcresults_data(args.tournament_year)
    for _ in range(args.num_simulations):
        t = Tournament(args.tournament_year)
        t.play_group_stage(model)
        t.play_knockout_stages(model)
        total_loss = 0
        for team in teams:
            result = t.get_furthest_position_for_team(team)
            team_results[team][result] += 1
            if args.tournament_year != "2022":
                actual_result = wcresults_df.loc[
                    wcresults_df.Team == team].Stage.values[0]
                loss = get_difference_in_stages(result, actual_result)
                total_loss += loss
        loss_values.append(total_loss)
    team_records = [{"team": k, **v} for k,v in team_results.items()]
    df = pd.DataFrame(team_records)
    df.to_csv(args.output_csv)
    # output txt file containing loss function values
    if args.tournament_year != "2022":
        with open(args.output_loss_txt, "w") as outfile:
            for val in loss_values:
                outfile.write(f"{val}\n")


if __name__ == "__main__":
    main()
