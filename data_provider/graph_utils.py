import json
import os
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd


COUNTRY_REGION_FILES = {
    "Italy": "./dataset/Italy/italy_regions.csv",
    "Spain": "./dataset/Spain/spain_regions.csv",
    "England": "./dataset/England/england_regions.csv",
    "France": "./dataset/France/france_regions.csv",
    "California": "./dataset/California/california_regions.csv",
    "Massachusetts": "./dataset/Massachusetts/massachusetts_regions.csv",
    "NYC": "./dataset/NYCFlu/nyc_regions.csv",
}

COUNTRY_JSON_FILES = {
    "California": "./dataset/California/COVIDinCalifornia.json",
    "Massachusetts": "./dataset/Massachusetts/COVIDinMassachusetts.json",
    "NYC": "./dataset/NYCFlu/InfluenzaInNYC.json",
}


def infer_country(country=None, root_path=None, data_path=None):
    if country:
        return country

    search_text = os.path.normpath(os.path.join(root_path or "", data_path or "")).lower()
    for candidate in COUNTRY_REGION_FILES:
        if candidate.lower() in search_text:
            return candidate
    raise ValueError(
        "Unable to infer country from the provided paths. Pass `country` explicitly."
    )


def load_json_payload(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_json_path(country, root_path=None, data_path=None):
    if data_path and data_path.lower().endswith(".json"):
        return os.path.join(root_path or "", data_path)
    return COUNTRY_JSON_FILES.get(country)


def load_region_dataframe(country=None, root_path=None, data_path=None):
    country = infer_country(country=country, root_path=root_path, data_path=data_path)
    region_path = COUNTRY_REGION_FILES.get(country)
    if region_path is None:
        raise ValueError(f"Unsupported country `{country}`.")

    df = pd.read_csv(region_path)
    payload = None
    json_path = resolve_json_path(country, root_path=root_path, data_path=data_path)
    if json_path and os.path.exists(json_path):
        payload = load_json_payload(json_path)
        node_order = payload.get("node", [])
        if node_order:
            if "name" not in df.columns:
                raise ValueError(f"{region_path} must include a `name` column.")
            df = df.set_index("name")
            missing = [name for name in node_order if name not in df.index]
            if missing:
                raise ValueError(f"Missing regions in {region_path}: {missing}")
            df = df.loc[node_order].reset_index()
    return country, df, payload


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    earth_radius_km = 6371
    return c * earth_radius_km


def build_distance_matrix(region_df):
    n = len(region_df)
    distance_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            loc1 = region_df.iloc[i]
            loc2 = region_df.iloc[j]
            distance_matrix[i, j] = haversine(
                loc1["longitude"],
                loc1["latitude"],
                loc2["longitude"],
                loc2["latitude"],
            )
    return distance_matrix


def build_distance_weight_matrix(region_df, epsilon=1e-6):
    distance_matrix = build_distance_matrix(region_df)
    return 1.0 / (1.0 + distance_matrix + epsilon)


def get_population_vector(region_df, payload=None):
    if payload is not None and payload.get("population"):
        population = np.asarray(payload["population"], dtype=float)
        if population.shape[0] != len(region_df):
            raise ValueError("Population length does not match node count.")
        return population

    for column in ("population", "pop", "Population", "POPULATION"):
        if column in region_df.columns:
            return region_df[column].astype(float).to_numpy()

    raise ValueError("Population data is required to build a gravity network.")


def build_gravity_weight_matrix(
    region_df,
    population,
    alpha=1.0,
    beta=1.0,
    gamma=2.0,
    epsilon=1e-6,
):
    population = np.asarray(population, dtype=float)
    if population.shape[0] != len(region_df):
        raise ValueError("Population length does not match the number of regions.")

    distance_matrix = build_distance_matrix(region_df)
    safe_distance = np.maximum(distance_matrix, epsilon)
    mass_out = np.power(population[:, None], alpha)
    mass_in = np.power(population[None, :], beta)
    gravity = (mass_out * mass_in) / np.power(safe_distance, gamma)
    np.fill_diagonal(gravity, 0.0)
    return gravity


def adjacency_from_payload(payload):
    nodes = payload.get("node", [])
    if "A" in payload and payload["A"]:
        adj = np.asarray(payload["A"], dtype=float)
    else:
        adj = np.zeros((len(nodes), len(nodes)), dtype=float)
        for (src, dst), weight in zip(payload.get("G", []), payload.get("flow", [])):
            adj[int(src), int(dst)] = float(weight)
    return adj


def normalize_adjacency_matrix(adj, add_self_loops=True, scale_by_max=True, epsilon=1e-10):
    adj = np.asarray(adj, dtype=float).copy()
    if scale_by_max:
        max_value = float(adj.max())
        if max_value > 0:
            adj = adj / max_value

    if add_self_loops:
        adj = adj + np.eye(adj.shape[0], dtype=float)

    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree + epsilon), 0.0)
    return degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]
