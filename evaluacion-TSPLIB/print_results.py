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
from hydra.utils import get_original_cwd
from neuralsa.configs import NeuralSAExperiment
from scripts.optimo import obtener_optimo_tsplib

def load(path: str, name: str) -> Dict:
    with open(os.path.join(path, name + ".pkl"), "rb") as f:
        return pickle.load(f)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")

@hydra.main(config_path="scripts/conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:
    print("="*40)
    print(f"RESULTADOS PARA TEMPERATURA INICIAL {cfg.sa.init_temp}")
    print("="*40)
    
    # Imprimir resultados optimos TSPLIB
    tsp_file = os.path.join(get_original_cwd(), "datasets/pr76.tsp")
    tour_file = os.path.join(get_original_cwd(), "datasets/pr76.opt.tour")
    tour, total_cost = obtener_optimo_tsplib(tsp_file, tour_file)
    
    print("\n--- RESULTADO TOUR ÓPTIMO TSPLIB ---")
    print(f"FO (costo total del tour): {total_cost:.4f}")
    print("Tour óptimo (orden de nodos, 1-based):")
    tour_1_based = [int(x) + 1 for x in tour]
    print(tour_1_based)
    print("-"*40)

    # Load results
    path = os.path.join(os.getcwd(), cfg.results_path, cfg.problem)
    random_out = load(path, f"random_out_{cfg.problem_dim}-{cfg.training.method}-{cfg.sa.init_temp}")
    train_out_sampled = load(
        path,
        f"train_out_sampled_{cfg.problem_dim}-{cfg.training.method}-{cfg.sa.init_temp}",
    )

    # Print header
    header = "{:>8}  {:>4}  {:>24}  {:>12}".format("MODE", "K", "COST", "TIME")

    for m in [1, 2, 5, 10]:  # Defines the number of steps: K = m * problem_dim
        # Accumulators
        r, s = np.zeros(5), np.zeros(5)
        rt, st = np.zeros(5), np.zeros(5)
        r_tour, s_tour, = [None]*5, [None]*5

        for i in [1, 2, 3, 4, 5]:  # Different runs
            # Accumulate results
            r[i - 1] = torch.mean(random_out[m, i]["min_cost"])
            rt[i - 1] = random_out[m, i]["time"]
            r_tour[i-1] = random_out[m, i]["tour"]
            s[i - 1] = torch.mean(train_out_sampled[m, i]["min_cost"])
            st[i - 1] = train_out_sampled[m, i]["time"]
            s_tour[i - 1] = train_out_sampled[m, i]["tour"]

        # Print out mean results across the 5 runs
        # Print results for SA
        random_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
            "Random",
            str(m) + "x",
            str(np.round(np.mean(r), 3)) + " +- " + str(np.round(np.std(r), 3)),
            "{}".format(timedelta(seconds=int(np.mean(rt)))),
        )
        print(f"\n--- Random (m={m}) ---")
        print(header)
        print(random_res)
        error_relativo = (np.mean(r) - total_cost) / total_cost * 100
        print(f"Error relativo respecto al óptimo TSPLIB: {error_relativo:.2f}%")
        print(f"--- Tours ---")
        for i in range(5):
            # Convertir array 3D a lista de enteros
            tour_list = r_tour[i].flatten().astype(int).tolist()
            print(f"Run {i+1}: {tour_list}")
            
        # Print results for Sampled Neural SA
        sampled_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
            "Sampled",
            str(m) + "x",
            str(np.round(np.mean(s), 3)) + " +- " + str(np.round(np.std(s), 3)),
            "{}".format(timedelta(seconds=int(np.mean(st)))),
        )
        print(f"\n--- Sampled (m={m}) ---")
        print(header)
        print(sampled_res)
        error_relativo = (np.mean(s) - total_cost) / total_cost * 100
        print(f"Error relativo respecto al óptimo TSPLIB: {error_relativo:.2f}%")
        print(f"--- Tours ---")
        for i in range(5):
            # Convertir array 3D a lista de enteros
            tour_list = s_tour[i].flatten().astype(int).tolist()
            print(f"Run {i+1}: {tour_list}")


if __name__ == "__main__":
    main()
