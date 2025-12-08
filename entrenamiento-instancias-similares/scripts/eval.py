# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import os
import pickle
import time
import csv

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from neuralsa.configs import NeuralSAExperiment
from neuralsa.model import BinPackingActor, KnapsackActor, TSPActor
from neuralsa.problem import TSP, BinPacking, Knapsack
from neuralsa.sa import sa

# For reproducibility on GPU
torch.backends.cudnn.deterministic = True


def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")


def save(obj, path):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_to_csv(data_dict, path, method_name):
    """
    Save data dictionary to CSV file.
    
    Parameters
    ----------
    data_dict: dict
        Dictionary with structure {(m, i): {"costs": [...], "times": [...]}}
    path: str
        Path to save the CSV file
    method_name: str
        Name of the method (e.g., "random", "sampled", "greedy")
    """
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['method', 'm', 'i', 'step', 'cost', 'time'])
        
        # Write data
        for (m, i), values in data_dict.items():
            costs = values.get("costs", [])
            times = values.get("times", [])
            
            # Convert costs to list if it's a tensor
            if isinstance(costs, torch.Tensor):
                costs = costs.cpu().numpy().tolist()
            elif not isinstance(costs, list):
                costs = [costs]
            
            # Convert times to list if it's a tensor
            if isinstance(times, torch.Tensor):
                times = times.cpu().numpy().tolist()
            elif not isinstance(times, list):
                times = [times] if times is not None else []
            
            # Extract total time (times is a list with one element - total execution time)
            total_time = times[0] if len(times) > 0 else None
            if total_time is not None:
                if isinstance(total_time, torch.Tensor):
                    total_time = total_time.item()
            
            # Convert cost values to Python scalars
            cost_list = []
            for cost in costs:
                if isinstance(cost, torch.Tensor):
                    # Handle scalar tensors and multi-dimensional tensors
                    if cost.numel() == 1:
                        cost_list.append(cost.item())
                    else:
                        # If it's a multi-element tensor, take the mean or first element
                        cost_list.append(cost.mean().item() if cost.dim() > 0 else cost.item())
                else:
                    cost_list.append(float(cost) if cost is not None else None)
            
            # Write rows - one row per cost step, with total_time repeated for each step
            for step, cost_val in enumerate(cost_list):
                writer.writerow([method_name, m, i, step, cost_val, total_time])


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:
    if "cuda" in cfg.device and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA device not found. Running on cpu.")

    # Set Problem and Networks
    if cfg.problem == "knapsack":
        if cfg.capacity is None:
            # Set capacity as done in the paper
            if cfg.problem_dim == 50:
                cfg.capacity = 12.5
            elif cfg.problem_dim == 100:
                cfg.capacity = 25
            else:
                cfg.capacity = cfg.problem_dim / 8
        problem = Knapsack(cfg.problem_dim, device=cfg.device, params={"capacity": cfg.capacity})
        actor = KnapsackActor(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "binpacking":
        problem = BinPacking(cfg.problem_dim, device=cfg.device)
        actor = BinPackingActor(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "tsp":
        problem = TSP(cfg.problem_dim, device=cfg.device)
        actor = TSPActor(cfg.embed_dim, device=cfg.device)
    else:
        raise ValueError("Invalid problem name.")
    print(f"-> model path: {cfg.model_path}")

    if cfg.model_path is None:
        ## ACA DEFINIR QUÉ ACTOR SE QUIERE USAR. 
        # training_problem_dim = 20 es el entrenamiento aleatorio
        training_problem_dim = 40 if cfg.problem == "tsp" else 50
        cfg.model_path = (
            "models/" + cfg.problem + str(training_problem_dim) + "-" + cfg.training.method + ".pt"
            #ejemplo: models/tsp40-ppo.pt
        )

    # Load trained model
    actor.load_state_dict(torch.load(os.path.join(cfg.model_path), map_location=cfg.device))
    actor.eval()
    print("Loaded model at ", cfg.model_path)

    # Prefer CSV dataset if present, else use Kool's dataset (20/50/100), else random
    if cfg.problem == "tsp":
        csv_path = os.path.join(get_original_cwd(), cfg.data_path, "grafo_random_100.csv")
        if os.path.exists(csv_path):
            print(f"entreeeeee")
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(1, 2))  # X,Y
            N = data.shape[0]
            cfg.n_problems = 1
            cfg.problem_dim = N
            coords = torch.tensor(data, dtype=torch.float32, device=cfg.device).unsqueeze(0)  # [1, N, 2]
            problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
            problem.set_params(coords=coords)
        elif cfg.problem_dim in [20, 50, 100]:
            print(f"entre mal")
            filename = os.path.join(
                get_original_cwd(), cfg.data_path, "tsp" + str(cfg.problem_dim) + "_test_seed1234.pkl"
            )
            with open(filename, "rb") as f:
                tsp_test = pickle.load(f)

            cfg.n_problems = 10000
            coords = torch.tensor(tsp_test, device=cfg.device)
            problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
            problem.set_params(coords=coords)
        else:
            print(f"Entró no debería, no existe el archivo dic_nodos.csv")
            params = problem.generate_params(mode="test")
            params = {k: v.to(cfg.device) for k, v in params.items()}
            problem.set_params(**params)
    else:
        params = problem.generate_params(mode="test")
        params = {k: v.to(cfg.device) for k, v in params.items()}
        problem.set_params(**params)

    # Create accumulators
    # Store the minimum cost of each problem
    # Store the time taken to evaluate all instances
    train_out_greedy = {}  # Greedy Neural SA
    train_out_greedy2 = {}
    train_out_sampled = {}  # Sampled Neural SA
    train_out_sampled2 = {}
    random_out = {}  # Vanilla SA
    random_out2 = {}

    
    for m in [1, 2, 5, 10]:  # Defines the number of steps: K = m * problem_dim
        # for i in [1, 2, 3, 4, 5]:  # Different runs
        for i in [1]:
            # Set seeds
            torch.manual_seed(i)
            actor.manual_seed(i)

            cfg.sa.outer_steps = cfg.problem_dim * m
            # If TSP K = m * problem_dim^2
            if cfg.problem == "tsp":
                cfg.sa.outer_steps = cfg.sa.outer_steps * cfg.problem_dim

            # Define temperature decay parameter as a function of the number of steps
            alpha = np.log(cfg.sa.stop_temp) - np.log(cfg.sa.init_temp)
            cfg.sa.alpha = np.exp(alpha / cfg.sa.outer_steps).item()

            # Define initial solution
            init_x = problem.generate_init_x().to(cfg.device)

            # if m == 10:
                # Evaluate vanilla SA
            torch.cuda.empty_cache()
            start_time = time.time()
            out = sa(actor, problem, init_x, cfg, replay=None, baseline=True, record_state=True)
            random_out[m, i] = {}
            random_out[m, i]["min_cost"] = out["min_cost"]
            random_out[m, i]["time"] = time.time() - start_time
            random_out[m,i]["tour"] = out["best_x"].cpu().numpy()
            random_out2[m, i] = {}
            random_out2[m, i]["costs"] = out["costs"]
            random_out2[m, i]["times"] = out["times"]

            torch.cuda.empty_cache()
            start_time = time.time()
            out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False, record_state=True)
            train_out_sampled[m, i] = {}
            train_out_sampled[m, i]["min_cost"] = out["min_cost"]
            train_out_sampled[m, i]["time"] = time.time() - start_time
            train_out_sampled[m, i]["tour"] = out["best_x"].cpu().numpy()
            train_out_sampled2[m, i] = {}
            train_out_sampled2[m, i]["costs"] = out["costs"]
            train_out_sampled2[m, i]["times"] = out["times"]
            res = torch.mean(train_out_sampled[m, i]["min_cost"]).item()

            print(
                str(m) + "x,",
                "K=" + str(cfg.sa.outer_steps) + ",",
                "random seed",
                i,
                "sampled:",
                "{:0.2f}".format(res),
            )

            if m == 1:
                # Evaluate greedy Neural SA
                torch.cuda.empty_cache()
                start_time = time.time()
                out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=True, record_state=True)
                train_out_greedy[m, i] = {}
                # train_out_greedy[m, i]["min_cost"] = out["min_cost"]
                # train_out_greedy[m, i]["time"] = time.time() - start_time
                # train_out_greedy[m, i]["tour"] = out["best_x"].cpu().numpy()
                # train_out_greedy2[m, i] = {}
                # train_out_greedy2[m, i]["costs"] = out["costs"]
                # train_out_greedy2[m, i]["times"] = out["times"]

            # res = torch.mean(train_out_sampled[m, i]["min_cost"]).item()

            # print(
            #     str(m) + "x,",
            #     "K=" + str(cfg.sa.outer_steps) + ",",
            #     "random seed",
            #     i,
            #     "sampled:",
            #     "{:0.2f}".format(res),
            # )

    path = os.path.join(os.getcwd(), "results", cfg.problem)
    create_folder(path)
    save(
        random_out,
        os.path.join(path, "random_out_" + str(cfg.problem_dim) + "-" + cfg.training.method+ "-" + str(cfg.sa.init_temp)),
    )
    # save(
    #     random_out2,
    #     os.path.join(path, "random_out2_" + str(cfg.problem_dim) + "-" + cfg.training.method),
    # )
    save(
        train_out_sampled,
        os.path.join(path, "train_out_sampled_" + str(cfg.problem_dim) + "-" + cfg.training.method+ "-" + str(cfg.sa.init_temp)),
    )
    # save(
    #     train_out_sampled2,
    #     os.path.join(path, "train_out_sampled2_" + str(cfg.problem_dim) + "-" + cfg.training.method),
    # )
    save(
        train_out_greedy,
        os.path.join(path, "train_out_greedy_" + str(cfg.problem_dim) + "-" + cfg.training.method+ "-" + str(cfg.sa.init_temp)),
    )
    # save(
    #     train_out_greedy2,
    #     os.path.join(path, "train_out_greedy2_" + str(cfg.problem_dim) + "-" + cfg.training.method),
    # )
    
    # Save to CSV
    # if random_out2:
    #     save_to_csv(
    #         random_out2,
    #         os.path.join(path, "random_out2_" + str(cfg.problem_dim) + "-" + cfg.training.method + ".csv"),
    #         "random"
    #     )
    if random_out2:
        for (m, i) in random_out2.keys():
            single_run_data = {(m, i): random_out2[(m, i)]}
            save_to_csv(
                single_run_data,
                os.path.join(path, f"random_out2_{cfg.problem_dim}-{cfg.training.method}_temp{cfg.sa.init_temp}_m{m}_i{i}.csv"),
                "random"
            )

    # if train_out_sampled2:
    #     save_to_csv(
    #         train_out_sampled2,
    #         os.path.join(path, "train_out_sampled2_" + str(cfg.problem_dim) + "-" + cfg.training.method + ".csv"),
    #         "sampled"
    #     )
    if train_out_sampled2:
        for (m, i) in train_out_sampled2.keys():
            single_run_data = {(m, i): train_out_sampled2[(m, i)]}
            save_to_csv(
                single_run_data,
                os.path.join(path, f"train_out_sampled2_{cfg.problem_dim}-{cfg.training.method}_temp{cfg.sa.init_temp}_m{m}_i{i}.csv"),
                "sampled"
            )

    if train_out_greedy2:
        for (m, i) in train_out_greedy2.keys():
            single_run_data = {(m, i): train_out_greedy2[(m, i)]}
            save_to_csv(
                single_run_data,
                os.path.join(path, f"train_out_greedy2_{cfg.problem_dim}-{cfg.training.method}_temp{cfg.sa.init_temp}_m{m}_i{i}.csv"),
                "greedy"
            )


if __name__ == "__main__":
    main()
