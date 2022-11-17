import os
import argparse
import pandas as pd

from multiprocessing import Process, Queue

from wcpredictor.src.utils import get_and_train_model, forecast_evaluation
from .run_simulations import get_dates_from_years_training


def get_cmd_line_args():
    parser = argparse.ArgumentParser(description="scan hyperparameters")
    parser.add_argument(
        "--metric",
        help="which metric to use",
        choices=["brier", "rps"],
        default="rps",
    )
    parser.add_argument(
        "--years_training",
        help="comma-separated list of num-years-training-data",
        default="20",
    )
    parser.add_argument(
        "--years_testing",
        help="num-years-testing-data (must be less than years_training)",
        default="2",
    )
    parser.add_argument(
        "--ratings_choices",
        help="what rankings data to use - comma-separated list",
        choices=[
            "game",
            "org",
            "none",
            "game,org",
            "game,none",
            "org,none",
            "game,org,none",
        ],
        default="game,org,none",
    )
    parser.add_argument(
        "--exclude_friendlies", help="exclude friendlies", action="store_true"
    )
    parser.add_argument(
        "--epsilon_choices",
        help="what value of epsilon to choose in weightings - comma-separated list",
        default="0.05,0.1,0.2",
    )
    parser.add_argument(
        "--world_cup_weight_choices",
        help="how much to weight the World Cup games and other competitions - comma-separated list",
        default="2,5",
    )
    parser.add_argument(
        "--output_dir", help="where to put output", type=str, default="output"
    )
    parser.add_argument(
        "--num_thread", help="how many threads for multiprocessing", type=int, default=4
    )
    args = parser.parse_args()
    return args


def run_metrics_wrapper(queue, pid, output_dir):
    print("In run_metrics_wrapper")
    while True:
        status = queue.get()
        if status == "DONE":
            print(f"Process {pid} finished all jobs!")
            break
        (
            metric,
            num_years,
            train_start,
            train_end,
            test_start,
            test_end,
            ratings,
            comps,
            epsilon,
            wc_weight,
        ) = status

        if len(comps) == 6:
            comptxt = "all_comps"
        else:
            comptxt = "no_friendlies"

        wc_pred = get_and_train_model(
            start_date=train_start,
            end_date=train_end,
            competitions=comps,
            rankings_source=ratings,
            epsilon=epsilon,
            world_cup_weight=wc_weight,
        )

        metrics = forecast_evaluation(
            model=wc_pred.model,
            start_date=test_start,
            end_date=test_end,
            competitions=comps,
            method=metric,
        )

        metrics_filename = (
            f"{metric}_{num_years}_{ratings}_{comptxt}_ep_{epsilon}_wc_{wc_weight}.txt"
        )
        metrics_filename = os.path.join(output_dir, metrics_filename)

        with open(metrics_filename, "w") as outfile:
            for val in metrics:
                outfile.write(f"{val}\n")
        print(f"Process {pid} Wrote file {metrics_filename}")


def main():
    args = get_cmd_line_args()
    metric = args.metric
    train_years = args.years_training.split(",")
    test_years = args.years_testing
    if any([int(x) <= int(test_years) for x in train_years]):
        raise ValueError(
            "each year in years_training must be greater than years_testing"
        )
    ratings = args.ratings_choices.split(",")
    competitions = [["W", "WQ", "C1", "CQ", "C2", "F"]]
    if args.exclude_friendlies:
        competitions.append(["W", "WQ", "C1", "CQ", "C2"])
    epsilons = [float(x) for x in args.epsilon_choices.split(",")]
    wc_weights = [float(x) for x in args.world_cup_weight_choices.split(",")]
    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # first add items to our multiprocessing queue
    queue = Queue()
    for num_years in train_years:
        train_start, test_end = get_dates_from_years_training(2022, int(num_years))
        train_end = pd.Timestamp(test_end) - pd.DateOffset(
            years=int(test_years), days=1
        )
        test_start = pd.Timestamp(test_end) - pd.DateOffset(years=int(test_years))
        # convert to string
        train_end = str(train_end.date())
        test_start = str(test_start.date())
        print(test_start)
        print(test_end)
        for r in ratings:
            if r == "none":
                r = None
            for comps in competitions:
                for ep in epsilons:
                    for wc in wc_weights:
                        print("adding to queue")
                        queue.put(
                            (
                                metric,
                                num_years,
                                train_start,
                                train_end,
                                test_start,
                                test_end,
                                r,
                                comps,
                                ep,
                                wc,
                            )
                        )
                        pass  # end of loop over world cup weight choices
                    pass  # end of loop over epsilon choices
                pass  # end of loop over competitions to exclude
            pass  # end of loop over ratings method
        pass  # end of loop over num_years_training

    # add some items to the queue to make the target function exit
    for i in range(args.num_thread):
        queue.put("DONE")

    # define processes for running the jobs
    procs = []
    for i in range(args.num_thread):
        p = Process(
            target=run_metrics_wrapper,
            args=(queue, i, args.output_dir),
        )
        p.daemon = True
        p.start()
        procs.append(p)

    # finally start the processes
    for i in range(args.num_thread):
        procs[i].join()


if __name__ == "__main__":
    main()
