# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import os
import pickle
from datetime import timedelta
from typing import Dict

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore

from neuralsa.configs import NeuralSAExperiment


def load(path: str, name: str) -> Dict:
    with open(os.path.join(path, name + ".pkl"), "rb") as f:
        return pickle.load(f)


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:
    # Load results
    path = os.path.join(os.getcwd(), cfg.results_path, cfg.problem)
    random_out = load(path, "random_out_" + str(cfg.problem_dim) + "-" + cfg.training.method + "-" + str(1.0))
    train_out_sampled = load(
        path,
        "train_out_sampled_" + str(cfg.problem_dim) + "-" + cfg.training.method + "-" + str(1.0),
    )
    train_out_greedy = load(
        path, "train_out_greedy_" + str(cfg.problem_dim) + "-" + cfg.training.method + "-" + str(1.0)
    )

    # Print header
    header = "{:>8}  {:>4}  {:>24}  {:>12}".format("MODE", "K", "COST", "TIME")
    print(header)

    for m in [1, 2, 5, 10]:  # Defines the number of steps: K = m * problem_dim
        # Accumulators
        r, s, g = np.zeros(5), np.zeros(5), np.zeros(5)
        rt, st, gt = np.zeros(5), np.zeros(5), np.zeros(5)
        r_tour, s_tour, g_tour = [None]*5, [None]*5, [None]*5  # Listas para almacenar tours

        for i in [1, 2, 3, 4, 5]:  # Different runs
            # Accumulate results
            if (m, i) in random_out:
                r[i - 1] = torch.mean(random_out[m, i]["min_cost"])
                rt[i - 1] = random_out[m, i]["time"]
                if i == 1:
                    print(f"Debug - Keys en random_out[{m}, {i}]: {list(random_out[m, i].keys())}")
                if "tour" in random_out[m,i]:
                    r_tour[i-1] = random_out[m, i]["tour"]
            
            if (m, i) in train_out_sampled:
                s[i - 1] = torch.mean(train_out_sampled[m, i]["min_cost"])
                st[i - 1] = train_out_sampled[m, i]["time"]
                if "tour" in train_out_sampled[m, i]:
                    s_tour[i - 1] = train_out_sampled[m, i]["tour"]
                # s[i - 1] = torch.mean(train_out_sampled[m, i]["min_cost"])
                # st[i - 1] = train_out_sampled[m, i]["time"]
                # s_tour[i - 1] = train_out_sampled[m, i]["tour"]
            # if m == 1:
            #     g[i - 1] = torch.mean(train_out_greedy[m, i]["min_cost"])
            #     gt[i - 1] = train_out_greedy[m, i]["time"]

        # Print out mean results across the 5 runs
        # Print results for vanilla SA
        # if m == 10:
        random_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
            "Random",
            str(m) + "x",
            str(np.round(np.mean(r), 3)) + " +- " + str(np.round(np.std(r), 3)),
            "{}".format(timedelta(seconds=int(np.mean(rt)))),
        )
        print(random_res)
        print(f"\n--- Tours para Random (m={m}) ---")
        for i in range(5):
            if r_tour[i] is not None:
                # Convertir array 3D a lista de enteros
                tour_list = r_tour[i].flatten().astype(int).tolist()
                print(f"Run {i+1}: {tour_list}")

        # Print results for Sampled Neural SA

        # if m == 10:
        sampled_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
            "Sampled",
            str(m) + "x",
            str(np.round(np.mean(s), 3)) + " +- " + str(np.round(np.std(s), 3)),
            "{}".format(timedelta(seconds=int(np.mean(st)))),
        )
        print(sampled_res)
        print(f"\n--- Tours para Sampled (m={m}) ---")
        for i in range(5):
            if s_tour[i] is not None:
                # Convertir array 3D a lista de enteros
                tour_list = s_tour[i].flatten().astype(int).tolist()
                print(f"Run {i+1}: {tour_list}")

        # Print results for Greedy Neural SA
        # if m == 1:
        #     greedy_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
        #         "Greedy",
        #         str(m) + "x",
        #         str(np.round(np.mean(g), 3)) + " +- " + str(np.round(np.std(g), 3)),
        #         "{}".format(timedelta(seconds=int(np.mean(gt)))),
        #     )
        #     print(greedy_res)


if __name__ == "__main__":
    main()
