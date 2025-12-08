import tsplib95
import torch
import numpy as np
from neuralsa.problem import TSP

def obtener_optimo_tsplib(tsp_file, tour_file):
    # 1. Cargar coordenadas del TSPLIB
    problem = tsplib95.load(tsp_file)
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
    coords = torch.tensor(coords).unsqueeze(0)  # [1, n_cities, 2]

    # 2. Cargar el tour óptimo
    with open(tour_file, "r") as f:
        lines = f.readlines()
    start = lines.index("TOUR_SECTION\n") + 1
    tour = []
    for line in lines[start:]:
        n = int(line.strip())
        if n == -1:
            break
        tour.append(n - 1)  # TSPLIB usa 1-based, PyTorch usa 0-based

    tour_tensor = torch.tensor(tour).view(1, -1, 1)  # [1, n_cities, 1]

    # 3. Calcular el costo usando tu clase TSP
    tsp = TSP(dim=coords.shape[1], n_problems=1)
    tsp.set_params(coords=coords)
    cost = tsp.get_edge_lengths_in_tour(tour_tensor)
    total_cost = cost.sum().item()
    return tour, total_cost