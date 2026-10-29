import argparse
import json
import os
from math import radians, sin, cos, sqrt, asin

import numpy as np
import pandas as pd


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def build_adj(df):
    n = len(df)
    distance_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            loc1 = df.iloc[i]
            loc2 = df.iloc[j]
            distance_matrix[i, j] = haversine(
                loc1["longitude"], loc1["latitude"],
                loc2["longitude"], loc2["latitude"]
            )

    epsilon = 1e-6
    adj = 1 / (1 + distance_matrix + epsilon)
    adj = adj + np.eye(n)
    adj = adj / adj.max()

    degree_matrix = np.diag(np.sum(adj, axis=1))
    degree_matrix_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
    normalized_adj = degree_matrix_inv_sqrt @ adj @ degree_matrix_inv_sqrt
    return normalized_adj


def main():
    parser = argparse.ArgumentParser(
        description="Build Massachusetts adjacency matrix from lat/lon distances."
    )
    parser.add_argument(
        "--regions",
        default="dataset/Massachusetts/massachusetts_regions.csv",
        help="CSV with columns: name,longitude,latitude",
    )
    parser.add_argument(
        "--json",
        default="dataset/Massachusetts/COVIDinMassachusetts.json",
        help="JSON file used for node order alignment",
    )
    parser.add_argument(
        "--out",
        default="dataset/Massachusetts/massachusetts_adj.npy",
        help="Output path for adjacency matrix (.npy)",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path for adjacency matrix",
    )
    args = parser.parse_args()

    if not os.path.exists(args.regions):
        raise FileNotFoundError(f"Missing regions file: {args.regions}")
    df = pd.read_csv(args.regions)
    required_cols = {"name", "longitude", "latitude"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Regions file must include columns: {sorted(required_cols)}")

    if args.json and os.path.exists(args.json):
        with open(args.json, "r", encoding="utf-8") as f:
            node_order = json.load(f).get("node", [])
        df = df.set_index("name")
        missing = [name for name in node_order if name not in df.index]
        if missing:
            raise ValueError(f"Missing regions in regions file: {missing}")
        df = df.loc[node_order].reset_index()

    adj = build_adj(df)
    np.save(args.out, adj)
    if args.out_csv:
        np.savetxt(args.out_csv, adj, delimiter=",")
    print(f"Saved adjacency to {args.out} with shape {adj.shape}.")


if __name__ == "__main__":
    main()
