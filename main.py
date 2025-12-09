# src/main.py

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from opening_nn import nearest_neighbor_routes, print_routes_summary
from opening_savings import savings_routes, print_savings_routes
from utils_distance import distance_between_points


# ===== تنظیمات قابل تغییر =====

EXCEL_FILE_NAME = "Daten_Knoten.xlsx"

# همه روزهای سناریو S1
S1_SHEETS = ["S1_1", "S1_2", "S1_3", "S1_4", "S1_5"]

COLUMN_MAPPING = {
    "id": "id_house",
    "lat": "lat",
    "lon": "lon",
    "num_packages": "num_packages",
}

# ظرفیت یک Lieferfahrzeug
VEHICLE_CAPACITY = 250

# مختصات Depot
DEPOT_LAT = 52.2640589
DEPOT_LON = 10.5401095

# فقط برای شیت اول S1_1 نقشه را نشان بده
PLOT_FIRST_SHEET = True


# ===== توابع کمکی =====

def list_excel_sheets(excel_path: Path) -> None:
    xls = pd.ExcelFile(excel_path)
    print("=== Available sheets in Excel ===")
    for name in xls.sheet_names:
        print("-", name)


def load_orders_from_sheet(excel_path: Path, sheet_name: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # فقط برای چک کردن mapping
    print("\nColumns in sheet:", list(df.columns))
    print(df.head())

    for logical_name, col_name in COLUMN_MAPPING.items():
        if col_name not in df.columns:
            raise KeyError(
                f"Expected column '{col_name}' for '{logical_name}' not found in sheet '{sheet_name}'. "
                f"Available columns: {list(df.columns)}"
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
    orders: List[Dict[str, Any]]
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


# اختیاری: نمایش گرافیکی ساده مشتری‌ها + Depot
def plot_orders_and_depot(sheet_name: str, orders: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    xs = [o["lon"] for o in orders]
    ys = [o["lat"] for o in orders]

    plt.scatter(xs, ys, s=5)  # مشتری‌ها
    plt.scatter([DEPOT_LON], [DEPOT_LAT], s=50, marker="x")  # Depot

    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title(f"Orders in sheet {sheet_name}")
    plt.show()


# ===== نقطه ورود اصلی =====

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    excel_path = project_root / "data" / EXCEL_FILE_NAME

    print(f"Using Excel file: {excel_path}")
    list_excel_sheets(excel_path)

    results = []

    for idx, sheet in enumerate(S1_SHEETS):
        print(f"\n===========================\nLoading sheet: {sheet}")
        orders = load_orders_from_sheet(excel_path, sheet)
        print(f"Loaded {len(orders)} orders from sheet '{sheet}'.")

        if PLOT_FIRST_SHEET and idx == 0:
            plot_orders_and_depot(sheet_name=sheet, orders=orders)

        # --- Nearest Neighbor ---
        print("\n=== Running Nearest Neighbor construction ===")
        nn_routes = nearest_neighbor_routes(
            orders=orders,
            vehicle_capacity=VEHICLE_CAPACITY,
            depot_lat=DEPOT_LAT,
            depot_lon=DEPOT_LON,
        )
        nn_total_km = compute_total_distance_km(nn_routes, orders)
        print_routes_summary(nn_routes, label="NN")
        print(f"Total distance (NN): {nn_total_km:.2f} km")

        # --- Savings ---
        print("\n=== Running Savings construction ===")
        sav_routes = savings_routes(
            orders=orders,
            vehicle_capacity=VEHICLE_CAPACITY,
            depot_lat=DEPOT_LAT,
            depot_lon=DEPOT_LON,
        )
        sav_total_km = compute_total_distance_km(sav_routes, orders)
        print_savings_routes(sav_routes)
        print(f"Total distance (Savings): {sav_total_km:.2f} km")

        results.append(
            {
                "sheet": sheet,
                "n_orders": len(orders),
                "nn_routes": len(nn_routes),
                "nn_km": nn_total_km,
                "sav_routes": len(sav_routes),
                "sav_km": sav_total_km,
            }
        )

    # ===== چاپ جدول خلاصه =====
    print("\n\n===== SUMMARY FOR S1 (all 5 days) =====")
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
