import itertools
import json
import os
from datetime import datetime
from multiprocessing import Process, Queue

import jsonpickle
import numpy as np
from bpl import NeutralDixonColesMatchPredictorWC

from wcpredictor.src.utils import get_and_train_model, test_model


def run_wrapper(
    queue,
    pid,
    womens,
    train_start,
    train_end,
    test_start,
    test_end,
    competitions,
    rankings_source,
    model,
    test_with_weights,
    output_dir,
):
    print("In run_wrapper")
    while True:
        status = queue.get()
        if status == "DONE":
            print(f"Process {pid} finished all jobs!")
            break

        epsilon, world_cup_weight = status

        wc_pred = get_and_train_model(
            start_date=train_start,
            end_date=train_end,
            womens=womens,
            competitions=competitions,
            rankings_source=rankings_source,
            epsilon=epsilon,
            world_cup_weight=world_cup_weight,
            model=model,
        )

        # test_competitions = {
        #     "likelihood_W_to_F": ["W", "C1", "WQ", "CQ", "C2", "F"],
        #     "likelihood_W_to_C2": ["W", "C1", "WQ", "CQ", "C2"],
        #     "likelihood_W_to_CQ": ["W", "C1", "WQ", "CQ"],
        #     "likelihood_W_to_C1": ["W", "C1"],
        # }
        test_competitions = {
            "likelihood_W": ["W"],
        }
        if test_with_weights:
            test_epsilon = epsilon
            test_world_cup_weight = world_cup_weight
        else:
            test_epsilon = 0
            test_world_cup_weight = 1.0

        likelihood = {
            name: test_model(
                model=wc_pred.model,
                start_date=test_start,
                end_date=test_end,
                womens=womens,
                competitions=comps,
                epsilon=test_epsilon,
                world_cup_weight=test_world_cup_weight,
                train_end_date=train_end,
            )
            for name, comps in test_competitions.items()
        }
        filename = (
            f"{int(datetime.now().timestamp())}_"
            f"epsilon_{epsilon}_worldcupweight_{world_cup_weight}"
        )
        with open(os.path.join(output_dir, f"{filename}.model"), "w") as f:
            f.write(jsonpickle.encode(wc_pred))
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(f"epsilon,world_cup_weight,{','.join(likelihood)}\n")
            f.write(
                f"{epsilon},{world_cup_weight},"
                f"{','.join(np.char.mod('%f', list(likelihood.values())))}"
            )
        print(f"Process {pid} Wrote file {output_dir}/{filename}")


def main():
    womens = False
    train_start = "1994-1-1"  # 2018 WC: 1998-1-1, 2014 WC: 1994-1-1, 2010 WC: 1990-1-1
    train_end = "2014-6-10"  # 2018 WC: 2018-6-12, 2014 WC: 2014-6-10, 2010 WC: 2010-6-9
    test_start = "2014-6-11"  # 2018WC: 2018-6-13, 2014WC: 2014-6-11, 2010WC: 2010-6-10
    test_end = "2014-7-14"  # 2018 WC: 2018-7-16, 2014 WC: 2014-7-14, 2010 WC: 2010-7-12
    competitions = ["W", "C1", "WQ", "CQ", "C2", "F"]
    rankings_source = None
    epsilon = [0.6, 0.8, 1.0, 1.25, 1.5]
    world_cup_weight = [1, 2, 3, 4, 5, 6]
    model = NeutralDixonColesMatchPredictorWC()
    test_with_weights = False
    output_dir = f"likelihood_scan_{int(datetime.now().timestamp())}"
    num_thread = 12

    # create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/params.json", "w") as f:
        json.dump(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "competitions": competitions,
                "rankings_source": rankings_source,
                "epsilon": epsilon,
                "world_cup_weight": world_cup_weight,
                "model": str(model),
                "test_with_weights": test_with_weights,
                "output_dir": output_dir,
                "num_thread": num_thread,
            },
            f,
        )

    # first add items to our multiprocessing queue
    queue = Queue()
    for eps, wcw in itertools.product(epsilon, world_cup_weight):
        queue.put((eps, wcw))

    # add some items to the queue to make the target function exit
    for _ in range(num_thread):
        queue.put("DONE")

    # define processes for running the jobs
    procs = []
    for i in range(num_thread):
        p = Process(
            target=run_wrapper,
            args=(
                queue,
                i,
                womens,
                train_start,
                train_end,
                test_start,
                test_end,
                competitions,
                rankings_source,
                model,
                test_with_weights,
                output_dir,
            ),
        )
        p.daemon = True
        p.start()
        procs.append(p)

    # finally start the processes
    for i in range(num_thread):
        procs[i].join()


if __name__ == "__main__":
    main()
