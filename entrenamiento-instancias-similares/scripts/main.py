# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random

import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import glob

from neuralsa.configs import NeuralSAExperiment
from neuralsa.model import (
    BinPackingActor,
    BinPackingCritic,
    KnapsackActor,
    KnapsackCritic,
    TSPActor,
    TSPCritic,
)
from neuralsa.problem import TSP, BinPacking, Knapsack
from neuralsa.sa import sa
from neuralsa.training import EvolutionStrategies
from neuralsa.training.ppo import ppo
from neuralsa.training.replay import Replay

import pandas as pd
import glob

### DEFINIR TIPO DE ENTRENAMIENTO
## descomentar tipo de entrenamiento que se va a usar. SOLO UNO DESCOMENTADO
# tipo_training = "aleatoria"
tipo_training = "instancias_similares"

# For reproducibility on GPU
torch.backends.cudnn.deterministic = True

#Crea una carpeta si no existe(para guardar models)
def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")


# Entrenamiento con Evolution Strategies (ES)
def train_es(actor, problem, init_x, es, cfg):
    with torch.no_grad():
        es.zero_updates()
        for _ in range(es.population):
            es.perturb(antithetic=True)

            # Run SA and compute the loss
            results = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
            loss = torch.mean(results[cfg.training.reward])
            es.collect(loss)

        es.step(reshape_fitness=True)

    return torch.mean(torch.tensor(es.objective))

# Entrenamiento con Proximal Policy Optimization (PPO)
def train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg):
    # Create replay to store transitions
    replay = Replay(cfg.sa.outer_steps * cfg.sa.inner_steps)
    # Run SA and collect transitions
    sa(actor, problem, init_x, cfg, replay=replay, baseline=False, greedy=False)
    # Optimize the policy with PPO
    ppo(actor, critic, replay, actor_opt, critic_opt, cfg)
#Idea: el actor decide movimientos/propuestas durante SA; se almacenan (estado, acción, recompensa…) en Replay; 
#luego PPO optimiza actor y critic con esas trayectorias.

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:

    if "cuda" in cfg.device and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA device not found. Running on cpu.")

    # Define temperature decay parameter as a function of the number of steps
    alpha = np.log(cfg.sa.stop_temp) - np.log(cfg.sa.init_temp)
    cfg.sa.alpha = np.exp(alpha / cfg.sa.outer_steps).item()

    print(OmegaConf.to_yaml(cfg))

    # Set seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set Problem and Networks
    if cfg.problem == "knapsack":
        problem = Knapsack(
            cfg.problem_dim, cfg.n_problems, device=cfg.device, params={"capacity": cfg.capacity}
        )
        actor = KnapsackActor(cfg.embed_dim, device=cfg.device)
        critic = KnapsackCritic(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "binpacking":
        problem = BinPacking(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        actor = BinPackingActor(cfg.embed_dim, device=cfg.device)
        critic = BinPackingCritic(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "tsp":
        problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        actor = TSPActor(cfg.embed_dim, device=cfg.device)
        critic = TSPCritic(cfg.embed_dim, device=cfg.device)
    else:
        raise ValueError("Invalid problem name.")

    # Set problem seed
    problem.manual_seed(cfg.seed)

    # If using PPO, initialize optimisers and replay
    if cfg.training.method == "ppo":
        actor_opt = torch.optim.Adam(
            actor.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        critic_opt = torch.optim.Adam(
            critic.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    elif cfg.training.method == "es":
        # Optimization specs
        optimizer = SGD(actor.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum)
        es = EvolutionStrategies(optimizer, cfg.training.stddev, cfg.training.population)
        milestones = [int(cfg.training.n_epochs * m) for m in cfg.training.milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        raise ValueError("Invalid training method.")

    with tqdm(range(cfg.training.n_epochs)) as t:
        if tipo_training == "aleatoria":
            print("====== ENTRENANDO CON INSTANCIAS ALEATORIAS =======")
            cfg.problem_dim = 40 #cambiar dirección del problema si se quiere otra dimensión
            print(f"Usando {cfg.n_problems} instancias aleatorias por época")
            print(f"Dimensión del problema: {cfg.problem_dim}")
            for i in t:
                # Create random instances
                params = problem.generate_params()
                params = {k: v.to(cfg.device) for k, v in params.items()}
                problem.set_params(**params)
                # Find initial solutions
                init_x = problem.generate_init_x()
                actor.manual_seed(cfg.seed)

                # Training loop
                if cfg.training.method == "ppo":
                    train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg)
                elif cfg.training.method == "es":
                    train_es(actor, problem, init_x, es, cfg)
                    scheduler.step()

                # Rerun trained model
                train_out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
                train_loss = torch.mean(train_out["min_cost"])

                t.set_description(f"Training loss: {train_loss:.4f}")

                path = os.path.join(os.getcwd(), "models")
                name = cfg.problem + str(cfg.problem_dim) + "-" + cfg.training.method + ".pt"
                create_folder(path)
                torch.save(actor.state_dict(), os.path.join(path, name))
        elif tipo_training == "instancias_similares":
            print("====== ENTRENANDO CON INSTANCIAS SIMILARES =======")
            # Cargar grafo_huge.csv
            huge_csv_path = os.path.join(get_original_cwd(), "training_datasets", "grafo_huge.csv")
            if not os.path.exists(huge_csv_path):
                raise FileNotFoundError(f"No se encontró el archivo: {huge_csv_path}")
            # cargar las coordenadas del archivo
            try:
                huge_coords = np.loadtxt(huge_csv_path, delimiter=",", skiprows=1, usecols=(1, 2), dtype=np.float32)
            except Exception:
                try:
                    huge_coords = np.loadtxt(huge_csv_path, usecols=(1, 2), dtype=np.float32)
                except Exception as e:
                    raise RuntimeError(f"Error al cargar {huge_csv_path}: {e}")
            if huge_coords.ndim != 2 or huge_coords.shape[1] != 2:
                raise RuntimeError(f"Formato inválido en {huge_csv_path}. Se esperaba [N, 2], se obtuvo {huge_coords.shape}")
            
            # Convertir a tensor y mover a device
            huge_coords_tensor = torch.from_numpy(huge_coords).to(cfg.device)  # [1000, 2]
            print(f"Cargado grafo_huge.csv con {huge_coords_tensor.shape[0]} nodos")

            # Configurar para crear 256 instancias (como dice el archivo configs.py) de dimensión 40
            N = 40  # Dimensión de cada instancia
            B = 256 # Número de instancias
            cfg.problem_dim = N
            cfg.n_problems = B

            # Recrear problema para el tamaño correcto
            if cfg.problem == "tsp":
                problem = TSP(N, B, device=cfg.device)
            elif cfg.problem == "binpacking":
                problem = BinPacking(N, B, device=cfg.device)
            elif cfg.problem == "knapsack":
                problem = Knapsack(N, B, device=cfg.device, params={"capacity": cfg.capacity})
            else:
                raise ValueError("Invalid problem name.")
            
            print(f"Entrenando con {B} instancias de dimensión {N} por época")

            for i in t:
                # Generar B instancias de tamaño N seleccionando aleatoriamente del pool de 1000 nodos
                # Usar generate_params con list_coords para muestrear sin reemplazo
                params = problem.generate_params(list_coords=huge_coords_tensor.cpu().numpy())
                params = {k: v.to(cfg.device) for k, v in params.items()}
                problem.set_params(**params)

                # Find initial solutions
                init_x = problem.generate_init_x()
                actor.manual_seed(cfg.seed)

                # Training loop
                if cfg.training.method == "ppo":
                    train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg)
                elif cfg.training.method == "es":
                    train_es(actor, problem, init_x, es, cfg)
                    scheduler.step()

                # Rerun trained model
                train_out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
                train_loss = torch.mean(train_out["min_cost"])

                t.set_description(f"Training loss: {train_loss:.4f}")

                path = os.path.join(os.getcwd(), "models")
                name = cfg.problem + str(cfg.problem_dim) + "-" + cfg.training.method + ".pt"
                create_folder(path)
                torch.save(actor.state_dict(), os.path.join(path, name))



                """
                # Obtener los archivos de entrenamiento y agrupamos por numero de nodos (N)
                datasets_root = os.path.join(get_original_cwd(), "training_datasets")
                csv_files = sorted(glob.glob(os.path.join(datasets_root, "grafo_*")))
                n_to_coords_list = {}
                for fp in csv_files:
                    try:
                        arr = np.loadtxt(fp, delimiter=",", skiprows=1, usecols=(1, 2), dtype=np.float32)
                    except Exception:
                        try:
                            arr = np.loadtxt(fp, usecols=(1, 2), dtype=np.float32)
                        except Exception:
                            continue
                    if arr.ndim != 2 or arr.shape[1] != 2:
                        continue
                    N = arr.shape[0]
                    n_to_coords_list.setdefault(N, []).append(torch.from_numpy(arr))

                if not n_to_coords_list:
                    raise RuntimeError("No se encontraron datasets válidos en training_datasets/")
                # Usar todas las instancias disponibles en training_datasets (misma dimensión)
                # Seleccionar el primer grupo disponible (todas tienen la misma dimensión)
                unique_N = sorted(n_to_coords_list.keys())
                if len(unique_N) != 1:
                    raise RuntimeError("Se esperaban datasets con la misma dimensión de nodos, pero se encontraron tamaños diferentes: {}".format(unique_N))
                N = unique_N[0]
                coords_pool = n_to_coords_list[N]
                B = len(coords_pool)
                coords = torch.stack(coords_pool, dim=0).to(cfg.device)  # [B, N, 2]

                # # elegimos un batch de datasets (round-robin por época)
                # unique_N = sorted(n_to_coords_list.keys())
                # bucket_idx = i % len(unique_N)
                # N = unique_N[bucket_idx]
                # coords_pool = n_to_coords_list[N]

                # # construir batch hasta cfg.n_problems (el minimo entre el numero de problemas y el numero de datasets)
                # # luego, seleccionamos los datasets y los convertimos a tensor
                # B = min(cfg.n_problems, len(coords_pool))
                # selected = coords_pool[:B]
                # coords = torch.stack(selected, dim=0).to(cfg.device)  # [B, N, 2] float32

                # Recrear problema para el numero de nodos N y el batch size B
                if cfg.problem == "tsp":
                    problem = TSP(N, B, device=cfg.device)
                elif cfg.problem == "binpacking":
                    problem = BinPacking(N, B, device=cfg.device)
                elif cfg.problem == "knapsack":
                    problem = Knapsack(N, B, device=cfg.device, params={"capacity": cfg.capacity})
                else:
                    raise ValueError("Invalid problem name.")

                # Actualizar valores de cfg para reflejar el bucket actual
                cfg.problem_dim = N
                cfg.n_problems = B

                # Set fixed instances
                problem.set_params(coords=coords)
                # Find initial solutions
                init_x = problem.generate_init_x()
                actor.manual_seed(cfg.seed)

                # Training loop
                if cfg.training.method == "ppo":
                    train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg)
                elif cfg.training.method == "es":
                    train_es(actor, problem, init_x, es, cfg)
                    scheduler.step()

                # Rerun trained model
                train_out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
                train_loss = torch.mean(train_out["min_cost"])

                t.set_description(f"Training loss: {train_loss:.4f}")

                path = os.path.join(os.getcwd(), "models")
                name = cfg.problem + str(cfg.problem_dim) + "-" + cfg.training.method + ".pt"
                create_folder(path)
                torch.save(actor.state_dict(), os.path.join(path, name))
                """
        else:
            raise ValueError(f"TIPO DE ENTRENAMIENTO INVÁLIDO:{tipo_training}❌")


if __name__ == "__main__":
    main()
