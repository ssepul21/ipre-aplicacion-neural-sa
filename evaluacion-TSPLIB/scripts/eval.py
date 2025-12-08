# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import os
import pickle
import csv
import time
import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
import tsplib95
from neuralsa.configs import NeuralSAExperiment
from neuralsa.model import BinPackingActor, KnapsackActor, TSPActor
from neuralsa.problem import TSP, BinPacking, Knapsack
from neuralsa.sa import sa
    
# For reproducibility on GPU
torch.backends.cudnn.deterministic = True

def load_tsplib_coords(filepath):
    problem = tsplib95.load(filepath)
    coords = []
    for i in range(1, problem.dimension + 1):
        x, y = problem.node_coords[i]
        coords.append([x, y])
    coords = np.array(coords, dtype=np.float32)
    # Normaliza a [0,1]
    x_range = coords[:, 0].max() - coords[:, 0].min()  # Rango en x
    y_range = coords[:, 1].max() - coords[:, 1].min()  # Rango en y
    factor = max(x_range, y_range)
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / factor  # x normalizada
    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / factor  # y normalizada
    return coords

def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")

def save(obj, path):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save2(obj, base_path, method_name, temp, problem_dim, training_method):
    for key, data in obj.items():
        m, i = key
        filename = f"{base_path}_{method_name}_{problem_dim}-{training_method}_temp{temp}_m{m}_i{i}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'cost', 'time'])
            costs = data.get('costs', [])
            times = data.get('times', [])
            if costs and hasattr(costs[0], 'item'):
                costs = [c.item() for c in costs]
            for step, (cost, time_val) in enumerate(zip(costs, times)):
                writer.writerow([step, cost, time_val])

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

    if cfg.model_path is None:
        training_problem_dim = 20 if cfg.problem == "tsp" else 50
        cfg.model_path = (
            "models/" + cfg.problem + str(training_problem_dim) + "-" + cfg.training.method + ".pt"
        )

    # Load trained model
    actor.load_state_dict(torch.load(os.path.join(cfg.model_path), map_location=cfg.device))
    actor.eval()
    print("Loaded model at ", cfg.model_path)

    tsplib_instance_path = os.path.join(get_original_cwd(), cfg.data_path, "pr76.tsp")  # Cambia el nombre según tu archivo
    if cfg.problem == "tsp" and os.path.isfile(tsplib_instance_path):
        coords = load_tsplib_coords(tsplib_instance_path)
        coords = torch.tensor(coords, device=cfg.device).unsqueeze(0) 
        cfg.problem_dim = coords.shape[1]
        cfg.n_problems = 1
        problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        problem.set_params(coords=coords)

    else:
        # Use Kool's dataset for TSP 20, 50 and 100
        if cfg.problem == "tsp" and cfg.problem_dim in [20, 50, 100]:
            filename = os.path.join(
                get_original_cwd(), cfg.data_path, "tsp" + str(cfg.problem_dim) + "_test_seed1234.pkl"
            )
            with open(filename, "rb") as f:
                tsp_test = pickle.load(f)

            cfg.n_problems = 10000  # These datasets have 10K instances
            coords = torch.tensor(tsp_test, device=cfg.device)
            problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
            problem.set_params(coords=coords)

        else:
            # Create random instances
            params = problem.generate_params(mode="test")
            params = {k: v.to(cfg.device) for k, v in params.items()}
            problem.set_params(**params)

    # Create accumulators
    # Store the minimum cost of each problem
    # Store the time taken to evaluate all instances
    '''train_out_greedy = {}  # Greedy Neural SA
    train_out_greedy2 = {} '''
    train_out_sampled = {}  # Sampled Neural SA
    train_out_sampled2 = {} 
    random_out = {}  # Vanilla SA
    random_out2 = {}  

    for m in [1, 2, 5, 10]:  # Defines the number of steps: K = m * problem_dim
        for i in [1, 2, 3, 4, 5]:  # Different runs
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

            torch.cuda.empty_cache()
            start_time = time.time()
            out = sa(actor, problem, init_x, cfg, replay=None, baseline=True)
            random_out[m, i] = {}
            random_out2[m, i] = {}
            random_out[m, i]["min_cost"] = out["min_cost"]
            random_out[m, i]["time"] = time.time() - start_time
            random_out[m, i]["tour"] = out["best_x"].cpu().numpy()
            random_out2[m, i]["costs"] = [cost.item() if hasattr(cost, 'item') else cost for cost in out["costs"]]
            random_out2[m, i]["times"] = out["times"]
                
            torch.cuda.empty_cache()
            start_time = time.time()
            out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
            train_out_sampled[m, i] = {}
            train_out_sampled2[m, i] = {}
            train_out_sampled[m, i]["min_cost"] = out["min_cost"]
            train_out_sampled[m, i]["time"] = time.time() - start_time
            train_out_sampled[m, i]["tour"] = out["best_x"].cpu().numpy()
            train_out_sampled2[m, i]["costs"] = [cost.item() if hasattr(cost, 'item') else cost for cost in out["costs"]]
            train_out_sampled2[m, i]["times"] = out["times"]
            
            '''if m == 1:
                # Evaluate greedy Neural SA
                torch.cuda.empty_cache()
                start_time = time.time()
                out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=True)
                train_out_greedy[m, i] = {}
                train_out_greedy2[m, i] = {}
                train_out_greedy[m, i]["min_cost"] = out["min_cost"]
                train_out_greedy[m, i]["time"] = time.time() - start_time
                train_out_greedy[m, i]["tour"] = out["best_x"].cpu().numpy()
                train_out_greedy2[m, i]["costs"] = [cost.item() if hasattr(cost, 'item') else cost for cost in out["costs"]]
                train_out_greedy2[m, i]["times"] = out["times"]'''
                

            res = torch.mean(train_out_sampled[m, i]["min_cost"]).item()

            print(
                str(m) + "x,",
                "K=" + str(cfg.sa.outer_steps) + ",",
                "random seed",
                i,
                "sampled:",
                "{:0.2f}".format(res),
            )

    
    path = os.path.join(os.getcwd(), "results", cfg.problem)
    create_folder(path)
    # Guardar evolución temporal para cada método
    save2(
        random_out2, 
        os.path.join(path, "random_out"),
        "2",
        cfg.sa.init_temp,
        cfg.problem_dim,
        cfg.training.method
        )

    save2(
        train_out_sampled2, 
        os.path.join(path, "train_out_sampled"),
        "2", 
        cfg.sa.init_temp,
        cfg.problem_dim,
        cfg.training.method
        )

    # Resultados finales en pickle
    save(
        random_out,
        os.path.join(path, "random_out_" + str(cfg.problem_dim) + "-" + cfg.training.method + "-" + str(cfg.sa.init_temp)),
    )

    save(
        train_out_sampled,
        os.path.join(path, "train_out_sampled_" + str(cfg.problem_dim) + "-" + cfg.training.method + "-" + str(cfg.sa.init_temp)),
    )

    '''save(
        train_out_greedy,
        os.path.join(path, "train_out_greedy_" + str(cfg.problem_dim) + "-" + cfg.training.method),
    )'''


if __name__ == "__main__":
    main()
