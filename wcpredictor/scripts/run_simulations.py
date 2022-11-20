#!/usr/bin/env python
import argparse
import math
import os
import random
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from time import time
from uuid import uuid4

import pandas as pd

from wcpredictor import (
    Tournament,
    get_and_train_model,
    get_difference_in_stages,
    get_teams_data,
    get_wcresults_data,
)


def get_cmd_line_args():
    parser = argparse.ArgumentParser("Simulate multiple World Cups")
    parser.add_argument(
        "--num_simulations",
        help="How many simulations to run in total",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--per_tournament",
        help="How many samples to run per tournament",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_thread", help="How many simulations to run", type=int, default=1
    )
    parser.add_argument(
        "--tournament_year",
        help="Which world cup to simulate? 2014, 2018 or 2022",
        choices={"2014", "2018", "2022"},
        default="2022",
    )
    parser.add_argument("--training_data_start", help="earliest date for training data")
    parser.add_argument("--training_data_end", help="latest date for training data")
    parser.add_argument(
        "--years_training_data",
        help="how many years of training data, before tournament start",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--output_csv", help="Path to output CSV file", default="sim_results.csv"
    )
    parser.add_argument(
        "--output_loss_txt",
        help="Path to output txt file of loss function vals",
        default="sim_results_loss.txt",
    )
    parser.add_argument(
        "--dont_use_ratings",
        help="If set, model is fitted without using the Fifa rankings of each team",
        action="store_true",
    )
    parser.add_argument(
        "--ratings_source",
        choices=["game", "org", "both"],
        default="org",
        help=(
            "if 'game' use FIFA video game ratings for prior, if 'org', use FIFA "
            "organization ratings"
        ),
    )
    parser.add_argument(
        "--include_competitions",
        help="comma-separated list of competitions to include in training data",
        default="W,C1,WQ,CQ,C2,F",
    )
    parser.add_argument(
        "--exclude_competitions",
        help="comma-separated list of competitions to exclude from training data",
    )
    parser.add_argument(
        "--epsilon",
        help="how much to downweight games by in exponential time weighting",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--world_cup_weight",
        help="how much more to weight World Cup games in the data",
        type=float,
        default=1.0,
    )
    parser.add_argument("--seed", help="seed value for simulations", type=int)

    return parser.parse_args()


def get_dates_from_years_training(tournament_year, years):
    start_year = int(tournament_year) - years
    # always start at 1st June, to capture the summer tournament
    start_date = f"{start_year}-06-01"
    # end at 1st June if tournament year is 2014 or 2018, or 20th Nov for 2022
    if tournament_year == "2022":
        end_date = "2022-11-20"
    else:
        end_date = f"{tournament_year}-06-01"
    return start_date, end_date


def get_start_end_dates(args):
    """
    Based on the command line args, define what period of training data to use.
    """
    if args.training_data_start and args.training_data_end:
        start_date = args.training_data_start
        end_date = args.training_data_end
    elif args.years_training_data:
        start_date, end_date = get_dates_from_years_training(
            args.tournament_year, args.years_training_data
        )
    else:
        raise RuntimeError(
            "Need to provide either start_date and end_date, or years_training_data arguments"
        )
    print(f"Start/End dates for training data are {start_date}, {end_date}")
    return start_date, end_date


def merge_csv_outputs(output_csv):
    files = glob(f"*_{output_csv}")
    df = pd.concat(
        [
            pd.read_csv(f, usecols=["Team", "Group", "R16", "QF", "SF", "RU", "W"])
            for f in files
        ]
    )
    df = df.groupby("Team").sum()
    print(df)
    df.to_csv(output_csv)
    for f in files:
        os.remove(f)


def run_sims(tournament_year, num_simulations, model, output_csv, output_txt):
    runid = str(uuid4())
    t = Tournament(tournament_year, num_samples=num_simulations)
    t.play_group_stage(model)
    t.play_knockout_stages(model)
    t.count_stages()
    t.stage_counts.to_csv(f"{runid}_{output_csv}")

    if tournament_year != "2022":
        teams_df = get_teams_data(tournament_year)
        teams = list(teams_df.Team.values)
        wcresults_df = None
        wcresults_df = get_wcresults_data(tournament_year)
        total_loss = 0
        for team in teams:
            actual_result = wcresults_df.loc[wcresults_df.Team == team].Stage.values[0]
            loss = get_difference_in_stages(result, actual_result)
            total_loss += loss
        loss_values = [total_loss]
        # output txt file containing loss function values
        with open(f"{runid}_{output_txt}", "w") as outfile:
            for val in loss_values:
                outfile.write(f"{val}\n")

    return runid


def run_wrapper(args):
    tournament_year, num_simulations, model, output_csv, output_txt = args
    return run_sims(tournament_year, num_simulations, model, output_csv, output_txt)


def main():
    args = get_cmd_line_args()
    # use the fifa ratings as priors?
    ratings_src = None if args.dont_use_ratings else args.ratings_source
    # list of competitions to include
    comps = args.include_competitions.split(",")
    if args.exclude_competitions:
        exclude_comps = args.exclude_competitions.split(",")
        for comp in exclude_comps:
            comps.remove(comp)
    start_date, end_date = get_start_end_dates(args)
    timestamp = int(datetime.now().timestamp())
    output_csv = f"{timestamp}_{args.output_csv}"
    output_loss_txt = f"{timestamp}_{args.output_loss_txt}"
    print(
        f"""
Running simulations with
tournament_year: {args.tournament_year}
num_simulations: {args.num_simulations}
start_date: {start_date}
end_date: {end_date}
comps: {comps}
rankings: {ratings_src}
{output_csv}
    """
    )
    if args.seed:
        random.seed(args.seed)

    model_start = time()
    model = get_and_train_model(
        start_date=start_date,
        end_date=end_date,
        competitions=comps,
        rankings_source=ratings_src,
        epsilon=args.epsilon,
        world_cup_weight=args.world_cup_weight,
    )
    model_time = time() - model_start
    print(f"Model fit took {model_time:.2f}s")

    sim_start = time()
    n_tournaments = math.ceil(args.num_simulations / args.per_tournament)
    sim_args = (
        (args.tournament_year, args.per_tournament, model, output_csv, output_loss_txt)
        for _ in range(n_tournaments)
    )
    with Pool(args.num_thread) as p:
        p.imap_unordered(run_wrapper, sim_args)
        p.close()
        p.join()

    merge_csv_outputs(output_csv)

    print(f"Model fit took {model_time:.2f}s")
    sim_time = time() - sim_start
    per_tournament = sim_time / args.num_simulations
    print(
        f"{args.num_simulations} tournaments took {sim_time:.2f}s "
        f"({per_tournament:.2f}s per tournament).\n(100,000 tournaments would take "
        f"{per_tournament * 100000 / (60 * 60):.2f} hours)"
    )


if __name__ == "__main__":
    main()
