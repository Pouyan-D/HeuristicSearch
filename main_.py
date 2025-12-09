# src/main.py

from pathlib import Path
from typing import List, Dict, Any
from collections import deque

import pandas as pd

from opening_nn import nearest_neighbor_routes
from opening_savings import savings_routes
from utils_distance import distance_between_points

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import folium
from sklearn.mixture import GaussianMixture

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg

# ===== ??????? =====

EXCEL_FILE_NAME = "Daten_Knoten.xlsx"

SCENARIOS: Dict[str, List[str]] = {
    "S1": ["S1_1"] #, "S1_2", "S1_3", "S1_4", "S1_5"],
    #"S2": ["S2_1", "S2_2", "S2_3", "S2_4", "S2_5"],
    #"S3": ["S3_1", "S3_2", "S3_3", "S3_4", "S3_5"],
}

COLUMN_MAPPING = {
    "id": "id_house",
    "lat": "lat",
    "lon": "lon",
    "num_packages": "num_packages",
}

VEHICLE_CAPACITY = 250
DEPOT_LAT = 52.2640589
DEPOT_LON = 10.5401095

PLOT_FIRST_SHEET = False      # ????? ???? ??????????
VERBOSE_ROUTES = False        # route ?? ?? ??? ???

# ===== intial definitions =====

earth_radius_km = 6371

class Cluster:
    id_: int
    nodes: List
    cost: float

    def __init__(self, id_: int, nodes: List, cost: float):
        self.id_ = id_
        self.nodes = nodes
        self.cost = cost

def haversine(theta):
    return np.power(np.sin(theta/2),2)

def hav_theta(spherical1: List, spherical2: List):
    return haversine(np.radians(spherical2[0])-np.radians(spherical1[0]))+np.cos(np.radians(spherical1[0]))*np.cos(np.radians(spherical2[0]))*haversine(np.radians(spherical2[1])-np.radians(spherical1[1]))
#sperical[0] denotes the latitude and sperical[1] denotes the longitude
def dist(spherical1: List, spherical2: List):
    return 2*earth_radius_km*np.arcsin(np.sqrt(hav_theta(spherical1, spherical2)))

# ===== ????? ???? =====

def list_excel_sheets(excel_path: Path) -> None:
    xls = pd.ExcelFile(excel_path)
    print("=== Available sheets in Excel ===")
    for name in xls.sheet_names:
        print("-", name)


def load_orders_from_sheet(excel_path: Path, sheet_name: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    for logical_name, col_name in COLUMN_MAPPING.items():
        if col_name not in df.columns:
            raise KeyError(
                f"Expected column '{col_name}' for '{logical_name}' not found "
                f"in sheet '{sheet_name}'. Available columns: {list(df.columns)}"
            )

    orders: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        orders.append(
            {
                "id": int(row[COLUMN_MAPPING["id"]]),
                "lat": float(row[COLUMN_MAPPING["lat"]]),
                "lon": float(row[COLUMN_MAPPING["lon"]]),
                "num_packages": int(row[COLUMN_MAPPING["num_packages"]]),
            }
        )
    return orders


def compute_total_distance_km(
    routes: List[List[int]],
    orders: List[Dict[str, Any]],
) -> float:
    coord_by_id = {o["id"]: (o["lat"], o["lon"]) for o in orders}
    depot = (DEPOT_LAT, DEPOT_LON)

    total = 0.0
    for route in routes:
        if not route:
            continue
        prev_point = depot
        for cid in route:
            cur_point = coord_by_id[cid]
            total += distance_between_points(prev_point, cur_point)
            prev_point = cur_point
        total += distance_between_points(prev_point, depot)
    return total

def dist_route(route, houses, hub_id):
  houses_by_id = houses.set_index('id_house')
  tot_dist = 0
  if hub_id == 0:
    start_coord = [DEPOT_LAT, DEPOT_LON]
  else:
    start_coord = [hubs.loc[hub_id].loc['lat'], hubs.loc[hub_id].loc['lon']]
  start_point = start_coord
  for i in range(len(route)):
    tot_dist += dist(start_point, [houses_by_id.loc[route[i]].loc['lat'],houses_by_id.loc[route[i]].loc['lon']])
    start_point = [houses_by_id.loc[route[i]].loc['lat'],houses_by_id.loc[route[i]].loc['lon']]
  tot_dist += dist(start_coord, [houses_by_id.loc[route[len(route)-1]].loc['lat'],houses_by_id.loc[route[len(route)-1]].loc['lon']])
  return tot_dist


def plot_orders_and_depot(sheet_name: str, orders: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    xs = [o["lon"] for o in orders]
    ys = [o["lat"] for o in orders]

    plt.scatter(xs, ys, s=5)
    plt.scatter([DEPOT_LON], [DEPOT_LAT], s=50, marker="x")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title(f"Orders in sheet {sheet_name}")
    plt.show()

# ======= Gaussian Mixture Clustering =======
color_iter = itertools.cycle([
     "navy", "c", "cornflowerblue", "gold", "darkorange",
    "crimson", "darkgreen", "lime", "purple", "magenta",
    "teal", "olive", "sienna", "orchid", "deepskyblue",
    "firebrick", "darkviolet", "peru", "dodgerblue", "mediumseagreen",
    "lightcoral", "indigo", "chocolate", "royalblue", "chartreuse",
    "turquoise", "mediumvioletred", "khaki", "slateblue", "salmon",
    "tomato", "plum", "forestgreen", "lightseagreen", "mediumorchid",
    "darkslategray", "goldenrod", "cadetblue", "coral", "palevioletred"
])

def plot_results(X, Y_, means, covariances, index, title, type_cov):
    if type_cov == "full":
      splot = plt.subplot(2, 1, 1 + index)
      for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
          v, w = linalg.eigh(covar)
          v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
          u = w[0] / linalg.norm(w[0])
          # as the DP will not use every component it has access to
          # unless it needs it, we shouldn't plot the redundant
          # components.
          if not np.any(Y_ == i):
              continue
          plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

          # Plot an ellipse to show the Gaussian component
          angle = np.arctan(u[1] / u[0])
          angle = 180.0 * angle / np.pi  # convert to degrees
          ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
          ell.set_clip_box(splot.bbox)
          ell.set_alpha(0.5)
          splot.add_artist(ell)

      plt.xlim(52.18, 52.36)
      plt.ylim(10.40, 10.65)
      plt.xticks(())
      plt.yticks(())
      plt.title(title)

    if type_cov == "spherical":
      splot = plt.subplot(2, 1, 1 + index)
      for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
          covar_mat = np.array([[covar,0],[0, covar]])
          v, w = linalg.eigh(covar_mat)
          v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
          u = w[0] / linalg.norm(w[0])
          # as the DP will not use every component it has access to
          # unless it needs it, we shouldn't plot the redundant
          # components.
          if not np.any(Y_ == i):
              continue
          plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

          # Plot an ellipse to show the Gaussian component
          angle = np.arctan(u[1] / u[0])
          angle = 180.0 * angle / np.pi  # convert to degrees
          ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
          ell.set_clip_box(splot.bbox)
          ell.set_alpha(0.5)
          splot.add_artist(ell)

      plt.xlim(52.18, 52.36)
      plt.ylim(10.40, 10.65)
      plt.xticks(())
      plt.yticks(())
      plt.title(title)

def get_clustering_for_hubs(scen_houses, houses, hubs, n_components, plot = False):
    sub_list_dict = {}
    for i in range(n_components):
        sub_list_dict[i]=[]
    houses_coordinates = scen_houses.set_index('id_house').drop('num_packages', axis = 1)


    X = houses_coordinates[['lat', 'lon']].to_numpy()

    gmm = GaussianMixture(n_components = n_components, covariance_type='spherical') #'spherical' or 'full'
    gmm.fit(X)

    chosen_hubs = []
    temp_hubs = {}
    for mean in gmm.means_:
        for id_ in hubs.index:
            temp_hubs[id_]=dist([mean[0],mean[1]], [hubs.loc[id_]["lat"],hubs.loc[id_]["lon"]])
        sorted_temp_hubs = sorted(temp_hubs.items(), key = lambda hub: hub[1])
        chosen_hubs.append(sorted_temp_hubs[0][0])
        
    predicted = gmm.predict(X)
    for index in range(len(X)):
        sub_list_dict[int(predicted[index])].append(houses.iloc[index]['id_house']) 

    if plot:
        plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture", 'spherical')

        # Koordinaten der gew‰hlten Hubs holen
        hub_coords = hubs.loc[chosen_hubs][['lat', 'lon']].to_numpy()

        # Schwarze Punkte plotten
        plt.scatter(
            hub_coords[:, 0], hub_coords[:, 1],
            c='black',
            marker='X',      # grˆﬂer als 'x'
            s=150,           # Punktgrˆﬂe
            linewidths=2
        )

        plt.show()

    print(chosen_hubs)

    return chosen_hubs

# ===== Folium =====

def draw_map(houses, Route, hub_id, file_name):
    avg_location = houses[['lat', 'lon']].mean()
    map_braunschweig = folium.Map(location=avg_location, zoom_start=13)


    houses_by_id = houses.set_index('id_house')

    if hub_id == 0:
      hub_coordinate = (DEPOT_LAT, DEPOT_LON)
    else:
      hub_coordinate = (hubs.loc[hub_id].loc['lat'], hubs.loc[hub_id].loc['lon'])
    start_id = hub_id
    start_point = hub_coordinate
    for i in range(len(Route)):
      #marker = folium.Marker(location=(houses_by_id.loc[Route[i]].loc['lat'], houses_by_id.loc[Route[i]].loc['lon']),tooltip = Route[i])
      line = folium.PolyLine(
            locations=[start_point,
                      (houses_by_id.loc[Route[i]].loc['lat'], houses_by_id.loc[Route[i]].loc['lon'])],
            tooltip=f"{start_id} to {Route[i]}",
        )
      start_id = Route[i]
      start_point = (houses_by_id.loc[Route[i]].loc['lat'], houses_by_id.loc[Route[i]].loc['lon'])
      line.add_to(map_braunschweig)
  
    line = folium.PolyLine(
          locations=[start_point,
                    hub_coordinate],
          tooltip=f"{start_id} to {hub_id}",
      )
    line.add_to(map_braunschweig)
    # add elements to the map

    marker = folium.Marker(location=hub_coordinate,tooltip = "Hub Nr. "+str(hub_id))
    marker.add_to(map_braunschweig)

    map_braunschweig.save(file_name+".html")

# ===== meta heuristics =====
def exp_decrease(time: int, lambda_: float)-> float:
    return np.exp(-np.abs(time*lambda_))

def two_opt_move(route: List[int])-> List[int]:
    i,j = sorted(random.sample(range(len(route)), 2))
    new_route = route[:i]+route[i:j][::-1]+route[j:]

    return new_route

def three_opt_move(route):
    i,j,k = sorted(random.sample(range(len(route)), 3))
    
    candidates = [
        route[:i]+route[j:k]+route[i:j]+route[k:],
        route[:i]+route[j:k][::-1]+route[i:j]+route[k:],
        route[:i]+route[j:k]+route[i:j][::-1]+route[k:],
        route[:i]+route[j:k][::-1]+route[i:j][::-1]+route[k:],
        ]
    return random.choice(candidates)

def two_opt_move_local(route, max_swap=6):
    n = len(route)

    # Depot muss an Position 0 (und optional am Ende) sein
    assert route[0] == 1

    # Kandidatenpositionen: alles auﬂer Depot
    positions = list(range(1, n))
    if len(positions) < 3:
        return route[:]

    i = random.choice(positions)
    j = (i + random.randint(1, max_swap)) % (n-1)

    if i < j:
        # normales Segment
        new_route = route[:i] + route[i:j][::-1] + route[j:]
        return new_route

    else:

      new_route = route[:j] + route[j:i][::-1]+route[i:]
      return new_route

def three_opt_move_local(route: List[int], max_segment: int = 8) -> List[int]:
    """3-opt move: swap/reverse three nearby segments."""
    n = len(route)
    i = random.randrange(n)
    j = i + random.randint(1, min(max_segment, n-i))
    k = j + random.randint(1, min(max_segment, n-j))
    k = min(k, n)

    candidates = [
        route[:i] + route[j:k] + route[i:j] + route[k:],               # swap segments
        route[:i] + route[j:k][::-1] + route[i:j] + route[k:],        # reverse second
        route[:i] + route[j:k] + route[i:j][::-1] + route[k:],        # reverse first
        route[:i] + route[j:k][::-1] + route[i:j][::-1] + route[k:], # reverse both
    ]
    return random.choice(candidates)

def simmulated_annealing(Routes, initial_temperature, temp_decrease, lambda_, max_epochs, move, hub_id, houses, max_num_rejections):
    not_changed = 0
    epoch = 0
    while epoch <= max_epochs and not_changed<=max_num_rejections:
        epoch+=1
        temperature = np.maximum(np.abs(initial_temperature)*temp_decrease(epoch, lambda_), 1e-6)
        index = random.choice(range(len(Routes)))
        route = Routes[index]
        d_current_config = dist_route(route, houses, hub_id)
        p_route = move(route)
        d_proposed_config = dist_route(p_route, houses, hub_id)
        delta = (d_proposed_config-d_current_config)

        print("Current:", d_current_config)
        print("Proposed:", d_proposed_config)

        if delta <= 0:
            Routes[index] = p_route
            not_changed = 0
        else:
            P_acc = np.exp(np.maximum(-delta, -700)/temperature)
            print(P_acc)

            P_random = random.uniform(0,1)

            if P_random <= P_acc:
                Routes[index] = p_route
                not_changed = 0

            else:
                not_changed += 1

    return Routes


def VNS(local_search, Routes, max_epochs, max_num_rejections, initial_temperature, temp_decrease, lambda_, max_epochs_ls, moves, hub_id, houses, max_num_rejections_ls ):
    k_max = len(moves)
    not_changed = 0
    k=0
    while epoch <= max_epochs and not_changed<=max_num_rejections:
        if k > k_max:
            k = 0
        epoch+=1
        move = moves[k]
        index = random.choice(range(len(Routes)))
        route = Routes[index]
        route_shaken = move(move(move(route)))
        route_shaken = local_search(Routes = [route_shaken], initial_temperature = initial_temperature, temp_decrease = temp_decrease, lambda_ = lambda_,
                                    max_epochs = max_epochs_ls, move = two_opt_move, hub_id = hub_id, houses = houses, max_num_rejections = max_num_rejections_ls)

        if dist_route(route_shaken, houses, hub_id) < dist_route(route, houses, hub_id):
            Routes[index] = route_shaken
            k=0
            not_changed = 0
        else:
            k+=1
            not_changed += 1

    return Routes


# ===== genetic algorithms =====
def boltzmann_probs(fitness, T):
    f = np.array(fitness)
    f = f - f.min()
    w = np.exp(-f / T)
    return w / w.sum()

def mutate_hubs(hubs_list, all_hubs):
    new = hubs_list[:]
    r = random.random()

    if r < 0.33:
        # delete
        if len(new) > 1:
            new.pop(random.randrange(len(new)))
    elif r < 0.66:
        # add
        unused = [h for h in all_hubs.index if h not in new]
        if unused:
            new.append(random.choice(unused))
    else:
        # replace
        if new:
            i = random.randrange(len(new))
            unused = [h for h in all_hubs.index if h not in new]
            if unused:
                new[i] = random.choice(unused)

    return new
    
def two_point_crossover(p1, p2, all_hubs, min_hubs=1):
    L1, L2 = len(p1), len(p2)
    L = min(L1, L2)

    if L < 1:
        return repair_hubs(p1 if L1 < L2 else p2, all_hubs)

    i, j = sorted(random.sample(range(L), 2))

    child = p1[:i] + p2[i:j] + p1[j:]

    return repair_hubs(child, all_hubs, min_hubs)

def repair_hubs(child, all_hubs, min_hubs=1):
    used = set()
    clean = []
    for h in child:
        if h not in used:
            clean.append(h)
            used.add(h)

    # falls zu wenige Hubs ? auff¸llen
    if len(clean) < min_hubs:
        missing = min_hubs - len(clean)
        unused = [h for h in all_hubs if h not in used]
        extra = random.sample(unused, missing)
        clean.extend(extra)

    return clean    

def build_member(hub_list, scen_houses_by_id):
    member={}
    member[0]=[[], 0.0]
    for hub in hub_list:
        member[0][0].append(hub)
        member[0][1]+=dist([DEPOT_LAT, DEPOT_LON],[hubs.loc[hub]["lat"],hubs.loc[hub]["lon"]])
        member[hub]=[[], 0.0]

    for h in scen_houses_by_id.index:
        dist_list = []
        for hub in hub_list:

            hub_to_house_dist = dist([scen_houses_by_id.loc[h]["lat"], scen_houses_by_id.loc[h]["lon"]],[hubs.loc[hub]["lat"],hubs.loc[hub]["lon"]])
            dist_list.append([hub,h,hub_to_house_dist])
        sorted_dist_list = sorted(dist_list, key= lambda h: h[2])
        member[sorted_dist_list[0][0]][0].append(sorted_dist_list[0][1])
        member[sorted_dist_list[0][0]][1]+=sorted_dist_list[0][2]

    return member # {hub/depot:[[customers],dist]}



def boltzmann_selection(num_start_clustered_config, num_members_of_population, epochs,scen_houses, houses, hubs, cost):
    population = [] # list of dicts [{hub/dpeot: [customer list[], total distance d], ...}]
    scen_houses_by_id = scen_houses.set_index('id_house')
    num_amount_gmm_seeded = math.ceil(0.3*num_members_of_population)
    num_childs = math.ceil(num_members_of_population/6)
    num_elite = math.ceil(num_members_of_population*0.07)

    print(num_amount_gmm_seeded)

    for i in range(num_start_clustered_config):

        n_components = 2+i*5
        hub_list = get_clustering_for_hubs(scen_houses=scen_houses, houses=houses, hubs=hubs, n_components=n_components)
        print(hub_list)

        member = build_member(hub_list=hub_list, scen_houses_by_id=scen_houses_by_id)        

        population.append(member)
        print(member)
        print(population[0])

    fitnesses = []
    for m in population:
        fitness = 0.0
        for h in m:
            fitness += m[h][1]
        fitnesses.append(fitness)
    print(fitnesses)
      
    while len(population) < num_amount_gmm_seeded:
        probs = boltzmann_probs(fitnesses, 1000)
        print(probs)
        parent_idx = np.random.choice(len(population), p=probs)
        print(parent_idx)

        parent = population[parent_idx]
        hub_list = list(parent.keys())
        hub_list.remove(0)
        print(hub_list)


        #population.append(parent) # only for testing

        new_hubs = mutate_hubs(hub_list, hubs)
        print(new_hubs)
        print('start building new member')

        # build new member
        new_member = build_member(hub_list=new_hubs, scen_houses_by_id=scen_houses_by_id)

        population.append(new_member)
        fitness = 0.0
        for h in new_member:
            fitness += new_member[h][1]
        fitnesses.append(fitness)
        print(fitnesses)

    while len(population) < num_members_of_population:
        k = random.randint(1, 25)
    
        hub_list =  random.sample(list(hubs.index), k)
        member = build_member(hub_list, scen_houses_by_id=scen_houses_by_id)
        population.append(member)
        fitness = 0.0
        for h in member:
            fitness += member[h][1]
        fitnesses.append(fitness)
    print(len(population))
    print(fitnesses)

    for i in range(epochs):
        for k in range(num_childs):
            probs = boltzmann_probs(fitnesses, 1000)


            # two parents
            pidx1, pidx2 = np.random.choice(len(population), size=2, replace=False, p=probs)

            parent1 = population[pidx1]
            parent2 = population[pidx2]

            hubs1 = list(parent1.keys())
            hubs1.remove(0)
            hubs2 = list(parent2.keys())
            hubs2.remove(0)

            child_hubs = two_point_crossover(
                hubs1, hubs2,
                all_hubs=hubs,
            )

            child = build_member(hub_list=child_hubs, scen_houses_by_id=scen_houses_by_id)
            population.append(child)
            fitness = 0.0
            for h in child:
                fitness += child[h][1]
            fitnesses.append(fitness)
            
        current_population = population.copy()
        current_fitnesses = fitnesses.copy()
        fitnesses_np = np.array(current_fitnesses)
        elite_idx = np.argpartition(fitnesses_np, num_elite)[:num_elite]

        population = [current_population[i] for i in elite_idx]
        fitnesses  = [current_fitnesses[i] for i in elite_idx]

        # Elite-Indices to set for faster creation of remaining population and remaining fitnesses
        elite_idx_set = set(elite_idx)

        # Restpopulation
        remaining_population = [
            ind for j, ind in enumerate(current_population) if j not in elite_idx_set
        ]
        remaining_fitnesses = [
            fit for j, fit in enumerate(current_fitnesses) if j not in elite_idx_set
        ]
        del current_population
        del current_fitnesses
        del probs
        del idx
        
        
        while len(population) < num_members_of_population:  
            probs = boltzmann_probs(remaining_fitnesses, 1000)
            idx = np.random.choice(len(remaining_population), p=probs)

            population.append(remaining_population[idx])
            fitnesses.append(remaining_fitnesses[idx])

            remaining_population.pop(idx)
            remaining_fitnesses.pop(idx)
        


# ===== reactive tabu search =====

def fitness_of_member(member: Dict[Any, list]) -> float:
    return sum(member[h][1] for h in member)

def random_neighbor(hub_list: List[int], all_hubs: List[int]):
    
    n = len(hub_list)
    move_type = random.choice(['add', 'del', 'replace', 'swap'])

    if move_type == 'add':
        unused = [h for h in all_hubs if h not in hub_list]
        if not unused:
            return hub_list[:], ('nop',)
        new = hub_list[:] + [random.choice(unused)]
        return new, ('add', new[-1])

    if move_type == 'del':
        if n <= 1:
            return hub_list[:], ('nop',)
        i = random.randrange(n)
        removed = hub_list[i]
        new = hub_list[:i] + hub_list[i+1:]
        return new, ('del', removed)

    if move_type == 'replace':
        if n == 0:
            return hub_list[:], ('nop',)
        i = random.randrange(n)
        old = hub_list[i]
        unused = [h for h in all_hubs if h not in hub_list]
        if not unused:
            return hub_list[:], ('nop',)
        new_h = random.choice(unused)
        new = hub_list[:]
        new[i] = new_h
        return new, ('replace', old, new_h)


def reactive_tabu_search(
    start_hubs: List[int],
    scen_houses,
    hubs_df,
    all_hubs: List[int],
    max_iter: int = 2000,
    neighborhood_size: int = 50,
    initial_tenure: int = 7,
    tenure_min: int = 20,
    tenure_max: int = 30,
    reactive_threshold: int = 100,
    no_improve_limit: int = 500,
    seed: int = None
):
    """
    Reactive Tabu Search for hub-location (single-solution search).

    Parameters
    ----------
    start_hubs : List[int]
        Initial hub IDs.
    scen_houses_by_id : DataFrame indexed by id_house (used by build_member_fn).
    hubs_df : DataFrame of candidate hubs (indexed by hub id).
    all_hubs : list of all candidate hub IDs.
    build_member_fn : callable(hub_list, scen_houses_by_id, hubs_df) -> member dict
    max_iter : int total iterations
    neighborhood_size : int number neighbors to sample per iter
    initial_tenure : int initial tabu tenure (will adapt)
    reactive_threshold : int iterations of stagnation to increase tenure
    no_improve_limit : global stopping if no improvement
    Returns best_member, best_hub_list, history (list of best fitnesses)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # initial solution
    scen_houses_by_id = scen_houses.set_index('id_house')
    delta1 = 1.2
    delta2 = 2

    cur_hubs = start_hubs[:]
    cur_member = build_member(hub_list = cur_hubs, scen_houses_by_id=scen_houses_by_id)  # {hub/depot:[[customers],dist]}
    cur_fit = fitness_of_member(cur_member)

    best_hubs = cur_hubs[:]
    best_member = cur_member
    best_fit = cur_fit

    visited = set()
    tabu_set = set()       
    tabu_queue = deque()   
    tenure = initial_tenure

    iter_total_no_improve = 0

    history = [best_fit]

    for it in range(1, max_iter + 1):
        # decrement tabu tenures
        if len(tabu_queue) > tenure:
            old = tabu_queue.popleft()
            tabu_set.remove(old)

        # generate sampled neighborhood (random neighbors)
        candidates = []
        for _ in range(neighborhood_size):
            neigh_hubs, move_key = random_neighbor(cur_hubs, all_hubs)
            # skip nop
            if move_key == ('nop',):
                continue
            # evaluate
            member = build_member(hub_list = cur_hubs, scen_houses_by_id=scen_houses_by_id)
            fit = fitness_of_member(member)
            candidates.append((fit, neigh_hubs, move_key, member))

        if not candidates:
            # nothing to do
            break

        # sort candidates by fitness ascending (minimization)
        candidates.sort(key=lambda x: x[0])

        # pick best admissible candidate, allowing aspiration if it beats best
        chosen = None
        for fit, neigh_hubs, move_key, member in candidates:
            is_tabu = move_key in tabu_set
            if (not is_tabu) or (fit < best_fit):  # aspiration
                chosen = (fit, neigh_hubs, move_key, member)
                break
            

        if chosen is None:
            # all tabu and none satisfy aspiration -> take best candidate and ignore tabu (fallback)
            chosen = candidates[0]

        fit, new_hubs, move_key, new_member = chosen

        solution_key = tuple(sorted(new_hubs))
        if solution_key in visited:          
            tenure= min(max(tenure*delta1 ,tenure+delta2) ,tenure_max)
        else: 
            tenure= max(min(tenure/delta1 ,tenure-delta2) ,tenure_min)
        
        visited.add(solution_key)

        # update current solution
        cur_hubs = new_hubs
        cur_member = new_member
        cur_fit = fit

        tabu_set.add(move_key)
        tabu_queue.append(move_key)

        # update best
        if cur_fit < best_fit:
            best_fit = cur_fit
            best_hubs = cur_hubs[:]
            best_member = cur_member
        
            iter_total_no_improve = 0
            # reactive: if improving, try reducing tenure slowly
            tenure = max(tenure_min, tenure - 1)
        else:
            iter_total_no_improve += 1

        history.append(best_fit)

        # global stopping if too long without improvement
        if iter_total_no_improve >= no_improve_limit:
            break

    return best_member, best_hubs, history

# ===== main =====

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    excel_path = project_root / "data" / EXCEL_FILE_NAME
    houses = pd.read_excel(excel_path, sheet_name='Houses')
    hubs = pd.read_excel(excel_path, sheet_name='Hubs').set_index('id_node')


    print(f"Using Excel file: {excel_path}")
    list_excel_sheets(excel_path)

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for scen_idx, (scen_name, sheets) in enumerate(SCENARIOS.items()):
        print(f"\n########## Scenario {scen_name} ##########")
        scenario_results: List[Dict[str, Any]] = []

        for idx, sheet in enumerate(sheets):
            sheet_data = pd.read_excel(excel_path, sheet_name=sheet)
            print(f"\n--- Loading sheet: {sheet} ---")
            orders = load_orders_from_sheet(excel_path, sheet)
            print(f"Loaded {len(orders)} orders from sheet '{sheet}'.")

            if PLOT_FIRST_SHEET and scen_idx == 0 and idx == 0:
                plot_orders_and_depot(sheet_name=sheet, orders=orders)

            # Nearest Neighbor
            nn_routes = nearest_neighbor_routes(
                orders=orders,
                vehicle_capacity=VEHICLE_CAPACITY,
                depot_lat=DEPOT_LAT,
                depot_lon=DEPOT_LON,
            )
            nn_total_km = compute_total_distance_km(nn_routes, orders)

            # Savings
            sav_routes = savings_routes(
                orders=orders,
                vehicle_capacity=VEHICLE_CAPACITY,
                depot_lat=DEPOT_LAT,
                depot_lon=DEPOT_LON,
            )
            sav_total_km = compute_total_distance_km(sav_routes, orders)

            if VERBOSE_ROUTES:
                print(f"NN: {len(nn_routes)} routes, {nn_total_km:.2f} km")
                print(f"Sav: {len(sav_routes)} routes, {sav_total_km:.2f} km")

            scenario_results.append(
                {
                    "sheet": sheet,
                    "n_orders": len(orders),
                    "nn_routes": len(nn_routes),
                    "nn_km": nn_total_km,
                    "sav_routes": len(sav_routes),
                    "sav_km": sav_total_km,
                }
            )

        all_results[scen_name] = scenario_results
        draw_map(houses, sav_routes[0], 0, 'map_'+sheet)
        print(dist_route(sav_routes[0], houses, 0))
        get_clustering_for_hubs(scen_houses=sheet_data, houses=houses, hubs=hubs, n_components=15, plot=True)


    # ===== ????? ????? ???? ?? ?????? =====
    for scen_name, results in all_results.items():
        print(f"\n\n===== SUMMARY for {scen_name} (5 days) =====")
        header = f"{'Sheet':<6} {'Orders':>7} {'NN_R':>6} {'NN_km':>12} {'Sav_R':>6} {'Sav_km':>12}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['sheet']:<6} "
                f"{r['n_orders']:>7d} "
                f"{r['nn_routes']:>6d} "
                f"{r['nn_km']:>12.2f} "
                f"{r['sav_routes']:>6d} "
                f"{r['sav_km']:>12.2f}"
            )
