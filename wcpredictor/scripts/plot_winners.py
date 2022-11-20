#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt
import pandas as pd

round_labels = {
    "Group": "Group stage",
    "R16": "Round of 16",
    "QF": "Quarter-final",
    "SF": "Semi-final",
    "RU": "Final",
    "W": "Winner",
}


def main():
    parser = argparse.ArgumentParser("plot the results of WC simulations")
    parser.add_argument("--input_csv", help="location of input file", required=True)
    parser.add_argument("--output_png", help="where to save image")
    parser.add_argument(
        "--counts", help="show counts rather than fraction if set", action="store_true"
    )
    parser.add_argument(
        "--num_teams", help="number of teams to plot", type=int, default=10
    )
    parser.add_argument(
        "--round",
        help="show progression to which round",
        choices={"Group", "R16", "QF", "SF", "RU", "W"},
        default="W",
    )
    parser.add_argument(
        "--exact_round",
        help=(
            "show count of knocked out at round specified rather"
            "than progression to at least that round if set"
        ),
        action="store_true",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    df.set_index("Team", inplace=True)
    if not args.counts:
        df = 100 * df.div(df.sum(axis=1), axis=0)
    if not args.exact_round:
        df = df[["W", "RU", "SF", "QF", "R16", "Group"]].cumsum(axis=1)
    df.sort_values(by=args.round, inplace=True)
    # plot the top num_teams values
    df = df[-args.num_teams :]
    fig, ax = plt.subplots(tight_layout=True)
    xvals = list(df.index)
    yvals = list(df[args.round].values)
    ax.barh(xvals, yvals)

    if not args.exact_round:
        xlabel = "Reached" if args.round != "W" else ""
    elif args.round == "RU":
        xlabel = ""
    else:
        xlabel = "Knocked out in" if args.round != "W" else ""
    if args.counts:
        ax.set_xlabel(f"{xlabel} {round_labels[args.round]} [count]")
    else:
        ax.set_xlabel(f"{xlabel} {round_labels[args.round]} [%]")
    if args.output_png:
        plt.savefig(args.output_png)
    else:
        plt.show()


if __name__ == "__main__":
    main()
