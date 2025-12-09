from typing import List, Dict, Any, Set, Tuple

from utils_distance import distance_between_points


def nearest_neighbor_routes(
    orders: List[Dict[str, Any]],
    vehicle_capacity: int,
    depot_lat: float,
    depot_lon: float,
) -> List[List[int]]:
    """
    ساخت routeها با روش Nearest Neighbor با در نظر گرفتن ظرفیت.
    خروجی: لیست routeها، هر route یک لیست از id_house (بدون Depot).
    """
    unserved: Set[int] = {o["id"] for o in orders}
    order_by_id: Dict[int, Dict[str, Any]] = {o["id"]: o for o in orders}

    routes: List[List[int]] = []
    depot = (depot_lat, depot_lon)

    while unserved:
        current_route: List[int] = []
        current_load = 0
        current_pos: Tuple[float, float] = depot

        while True:
            feasible = [
                cid for cid in unserved
                if current_load + order_by_id[cid]["num_packages"] <= vehicle_capacity
            ]
            if not feasible:
                break

            best_cid = None
            best_dist = float("inf")
            for cid in feasible:
                cand = order_by_id[cid]
                d = distance_between_points(
                    current_pos, (cand["lat"], cand["lon"])
                )
                if d < best_dist:
                    best_dist = d
                    best_cid = cid

            current_route.append(best_cid)
            current_load += order_by_id[best_cid]["num_packages"]
            current_pos = (order_by_id[best_cid]["lat"],
                           order_by_id[best_cid]["lon"])
            unserved.remove(best_cid)

        routes.append(current_route)

    return routes


def print_routes_summary(routes: List[List[int]], label: str = "NN") -> None:
    print(f"=== {label} Routes ===")
    for i, route in enumerate(routes, start=1):
        print(f"Route {i:02d}: {len(route)} stops, customers: {route}")
