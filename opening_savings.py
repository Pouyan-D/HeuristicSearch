# src/opening_savings.py

from typing import List, Dict, Any, Tuple

from utils_distance import distance_between_points


def _build_distance_cache(
    orders: List[Dict[str, Any]],
    depot_lat: float,
    depot_lon: float,
) -> Tuple[
    Dict[int, Tuple[float, float]],
    Dict[Tuple[int, int], float],
    Dict[int, float],
]:
    coord_by_id: Dict[int, Tuple[float, float]] = {
        o["id"]: (o["lat"], o["lon"]) for o in orders
    }

    depot = (depot_lat, depot_lon)

    dist_depot: Dict[int, float] = {}
    for cid, coord in coord_by_id.items():
        dist_depot[cid] = distance_between_points(depot, coord)

    ids = list(coord_by_id.keys())
    dist_ij: Dict[Tuple[int, int], float] = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ci, cj = ids[i], ids[j]
            d = distance_between_points(coord_by_id[ci], coord_by_id[cj])
            dist_ij[(ci, cj)] = d
            dist_ij[(cj, ci)] = d

    return coord_by_id, dist_ij, dist_depot


def savings_routes(
    orders: List[Dict[str, Any]],
    vehicle_capacity: int,
    depot_lat: float,
    depot_lon: float,
) -> List[List[int]]:
    demand = {o["id"]: o["num_packages"] for o in orders}
    if sum(demand.values()) == 0:
        return []

    coord_by_id, dist_ij, dist_depot = _build_distance_cache(
        orders, depot_lat, depot_lon
    )

    # هر مشتری اول تو route خودش
    routes: List[List[int]] = [[o["id"]] for o in orders]
    route_load: Dict[int, int] = {
        idx: demand[route[0]] for idx, route in enumerate(routes)
    }

    # mapping مشتری → index route
    customer_to_route: Dict[int, int] = {}
    for idx, route in enumerate(routes):
        for cid in route:
            customer_to_route[cid] = idx

    # لیست savings
    savings_list: List[Tuple[float, int, int]] = []
    ids = [o["id"] for o in orders]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i = ids[a]
            j = ids[b]
            s = dist_depot[i] + dist_depot[j] - dist_ij[(i, j)]
            # اگر saving منفی یا خیلی کوچک است، می‌تونی اینجا فیلتر کنی:
            # if s <= 0: continue
            savings_list.append((s, i, j))

    savings_list.sort(reverse=True, key=lambda x: x[0])

    def find_route_index(cid: int) -> int:
        return customer_to_route.get(cid, -1)

    for saving, i, j in savings_list:
        ri = find_route_index(i)
        rj = find_route_index(j)
        if ri == -1 or rj == -1 or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]
        if not route_i or not route_j:
            continue

        # i باید انتهای route_i و j ابتدای route_j باشد (یا برعکس)
        can_merge_ij = route_i[-1] == i and route_j[0] == j
        can_merge_ji = route_j[-1] == j and route_i[0] == i

        if not (can_merge_ij or can_merge_ji):
            continue

        new_load = route_load[ri] + route_load[rj]
        if new_load > vehicle_capacity:
            continue

        if can_merge_ij:
            new_route = route_i + route_j
        else:
            new_route = route_j + route_i

        # route جدید را در ri می‌نشانیم
        routes[ri] = new_route
        route_load[ri] = new_load

        # route_j را خالی می‌کنیم
        routes[rj] = []
        route_load.pop(rj, None)

        # mapping مشتری‌ها را آپدیت می‌کنیم
        for cid in new_route:
            customer_to_route[cid] = ri

    # فقط routeهای غیرخالی را برگردان
    final_routes = [r for r in routes if r]
    return final_routes


def print_savings_routes(routes: List[List[int]]) -> None:
    print("=== Savings Routes ===")
    for i, route in enumerate(routes, start=1):
        print(f"Route {i:02d}: {len(route)} stops, customers: {route}")
