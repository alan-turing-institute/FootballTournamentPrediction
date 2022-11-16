import os
import argparse

from multiprocessing import Process, Queue

from .run_simulations import run_sims, get_dates_from_years_training


def get_cmd_line_args():
    parser = argparse.ArgumentParser(description="scan hyperparameters")
    parser.add_argument(
        "--tournaments",
        help="comma-separated list of tournaments",
        choices=["2014", "2018", "2014,2018"],
        default="2018",
    )
    parser.add_argument(
        "--years_training",
        help="comma-separated list of num-years-training-data",
        default="4,5,6,7,8",
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
        "--num_simulations",
        help="how many tournaments per point",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output_dir", help="where to put output", type=str, default="output"
    )
    parser.add_argument(
        "--num_thread", help="how many threads for multiprocessing", type=int, default=4
    )
    args = parser.parse_args()
    return args


def run_sim_wrapper(queue, pid, num_simulations, output_dir):
    print("In run_sim_wrapper")
    while True:
        status = queue.get()
        if status == "DONE":
            print(f"Process {pid} finished all jobs!")
            break
        tournament, num_years, start_date, end_date, ratings, comps, epsilon, wc_weight = status

        if len(comps) == 6:
            comptxt = "all_comps"
        else:
            comptxt = "no_friendlies"
        csv_filename = f"{tournament}_{num_years}_{ratings}_{comptxt}_ep_{epsilon}_wc_{wc_weight}.csv"
        loss_filename = f"{tournament}_{num_years}_{ratings}_{comptxt}_ep_{epsilon}_wc_{wc_weight}_loss.txt"
        csv_filename = os.path.join(output_dir, csv_filename)
        loss_filename = os.path.join(output_dir, loss_filename)
        run_sims(
            tournament_year=tournament,
            num_simulations=num_simulations,
            start_date=start_date,
            end_date=end_date,
            competitions=comps,
            rankings_src=ratings,
            epsilon=epsilon,
            world_cup_weight=wc_weight,
            output_csv=csv_filename,
            output_txt=loss_filename,
        )
        print(f"Process {pid} Wrote file {csv_filename}")


def main():
    args = get_cmd_line_args()
    tournaments = args.tournaments.split(",")
    train_years = args.years_training.split(",")
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
    for tournament in tournaments:
        for num_years in train_years:
            start_date, end_date = get_dates_from_years_training(
                tournament, int(num_years)
            )
            for r in ratings:
                if r == "none":
                    r = None
                for comps in competitions:
                    for ep in epsilons:
                        for wc in wc_weights:        
                            print("adding to queue")
                            queue.put((tournament,
                                       num_years,
                                       start_date,
                                       end_date,
                                       r,
                                       comps,
                                       ep,
                                       wc))
                            pass # end of loop over world cup weight choices
                        pass # end of loop over epsilon choices
                    pass  # end of loop over competitions to exclude        
                pass  # end of loop over ratings method
            pass  # end of loop over num_years_training
        pass  # end of loop over tournaments

    # add some items to the queue to make the target function exit
    for i in range(args.num_thread):
        queue.put("DONE")

    # define processes for running the jobs
    procs = []
    for i in range(args.num_thread):
        p = Process(
            target=run_sim_wrapper,
            args=(queue, i, args.num_simulations, args.output_dir),
        )
        p.daemon = True
        p.start()
        procs.append(p)

    # finally start the processes
    for i in range(args.num_thread):
        procs[i].join()


if __name__ == "__main__":
    main()
