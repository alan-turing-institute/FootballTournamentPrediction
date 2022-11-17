#!/usr/bin/env python

import argparse

from wcpredictor import (
    get_and_train_model,
    get_fixture_data,
    get_most_probable_scoreline,
    predict_group_match,
    predict_knockout_match,
)


def get_cmd_line_args():
    parser = argparse.ArgumentParser("Print out match predictions")
    parser.add_argument(
        "--stage",
        help="What stage of the tournament?",
        choices=["G1", "G2", "G3", "R16", "QF", "SF", "F"],
        required=True,
    )
    parser.add_argument(
        "--sample", help="sample from the prob distribution?", action="store_true"
    )
    args = parser.parse_args()
    return args


def get_fixture_indices(stage):
    if stage == "G1":
        return 0, 16
    elif stage == "G2":
        return 16, 32
    elif stage == "G3":
        return 32, 48
    elif stage == "R16":
        return 48, 56
    elif stage == "QF":
        return 56, 60
    elif stage == "SF":
        return 60, 62
    elif stage == "F":
        return 62, 63
    else:
        raise RuntimeError("Unrecognized stage")


def main():
    args = get_cmd_line_args()
    model = get_and_train_model()
    fixture_indices = get_fixture_indices(args.stage)
    fixture_df = get_fixture_data()
    fixture_df = fixture_df[
        (fixture_df.index >= fixture_indices[0])
        & (fixture_df.index < fixture_indices[1])
    ]
    for _, row in fixture_df.iterrows():
        team_1 = row.Team_1
        team_2 = row.Team_2
        if args.sample:
            score_1, score_2 = predict_group_match(model, team_1, team_2)
            print(f"{team_1} {score_1}:{score_2} {team_2}")
        else:
            score_1, score_2, prob = get_most_probable_scoreline(model, team_1, team_2)
            print(
                f"{team_1} {score_1}:{score_2} {team_2} with probability of {prob*100:.1f}%"
            )


if __name__ == "__main__":
    main()
