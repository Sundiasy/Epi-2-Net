import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "This script needs pandas to read the Census movement workbook. "
        "Use the project env, e.g. `D:\\conda\\envs\\torch\\python.exe`."
    ) from exc


THIS_DIR = Path(__file__).resolve().parent
FLU_CSV = THIS_DIR / "ny_influenza_county_weekly.csv"
MIGRATION_XLSX = THIS_DIR / "county_to_county_migration_2011_2015.xlsx"
POP_CSV = THIS_DIR / "co-est2024-alldata.csv"
OUTPUT_JSON = THIS_DIR / "InfluenzaInNYC.json"
REGION_CSV = THIS_DIR / "nyc_regions.csv"


COUNTY_INFO = {
    "36005": {"name": "Bronx", "sheet_name": "Bronx County", "pop_name": "Bronx County"},
    "36047": {"name": "Kings", "sheet_name": "Kings County", "pop_name": "Kings County"},
    "36061": {"name": "New York", "sheet_name": "New York County", "pop_name": "New York County"},
    "36081": {"name": "Queens", "sheet_name": "Queens County", "pop_name": "Queens County"},
    "36085": {"name": "Richmond", "sheet_name": "Richmond County", "pop_name": "Richmond County"},
}

COORD_PATTERN = re.compile(r"\(([-0-9.]+),\s*([-0-9.]+)\)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an exact NYC county-level influenza dataset for Epi-2-Net."
    )
    parser.add_argument("--flu-csv", default=str(FLU_CSV))
    parser.add_argument("--migration-xlsx", default=str(MIGRATION_XLSX))
    parser.add_argument("--population-csv", default=str(POP_CSV))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--region-csv", default=str(REGION_CSV))
    parser.add_argument("--start-date", default=None, help="Inclusive start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date in YYYY-MM-DD.")
    return parser.parse_args()


def canonical_fips(state_code, county_code):
    return f"{int(state_code):02d}{int(county_code):03d}"


def parse_coordinates(raw_value):
    match = COORD_PATTERN.search(raw_value or "")
    if not match:
        raise ValueError(f"Could not parse coordinates from: {raw_value!r}")
    latitude = float(match.group(1))
    longitude = float(match.group(2))
    return longitude, latitude


def load_exact_flu_series(path):
    county_weekly = defaultdict(lambda: defaultdict(float))
    coordinates = {}
    weeks = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["region"] != "NYC":
                continue
            fips = row["fips"].strip()
            if fips not in COUNTY_INFO:
                continue

            week = row["weekendingdate"][:10]
            count = float(row["count"])
            county_weekly[fips][week] += count
            weeks.add(week)

            if fips not in coordinates:
                longitude, latitude = parse_coordinates(row["geocoded_column"])
                coordinates[fips] = (longitude, latitude)

    missing_coords = [fips for fips in COUNTY_INFO if fips not in coordinates]
    if missing_coords:
        raise ValueError(f"Missing coordinates for counties: {missing_coords}")

    return county_weekly, sorted(weeks), coordinates


def filter_weeks(weeks, start_date=None, end_date=None):
    filtered = []
    for week in weeks:
        if start_date and week < start_date:
            continue
        if end_date and week > end_date:
            continue
        filtered.append(week)
    if not filtered:
        raise ValueError(
            f"No weekly observations remain after filtering with start={start_date} end={end_date}."
        )
    return filtered


def load_population(path):
    population = {}
    with open(path, "r", encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["STATE"] != "36":
                continue
            fips = canonical_fips(row["STATE"], row["COUNTY"])
            if fips in COUNTY_INFO:
                population[fips] = int(row["POPESTIMATE2024"])

    missing = [fips for fips in COUNTY_INFO if fips not in population]
    if missing:
        raise ValueError(f"Missing population rows for counties: {missing}")
    return population


def load_exact_migration_matrix(path):
    df = pd.read_excel(path, sheet_name="New York", skiprows=3)

    current_state_col = "Unnamed: 0"
    current_county_col = "Unnamed: 1"
    previous_state_col = "Unnamed: 2"
    previous_county_col = "Unnamed: 3"
    current_name_col = "Unnamed: 5"
    previous_name_col = "Unnamed: 21"
    flow_col = "Estimate.14"

    borough_county_codes = {float(fips[-3:]) for fips in COUNTY_INFO}
    matrix = [[0.0 for _ in COUNTY_INFO] for _ in COUNTY_INFO]
    ordered_fips = list(COUNTY_INFO.keys())
    index_of = {fips: idx for idx, fips in enumerate(ordered_fips)}

    subset = df[
        (df[current_state_col] == "036")
        & (df[previous_state_col] == "036")
        & (df[current_county_col].isin(borough_county_codes))
        & (df[previous_county_col].isin(borough_county_codes))
    ]

    for _, row in subset.iterrows():
        dst_fips = canonical_fips(36, row[current_county_col])
        src_fips = canonical_fips(36, row[previous_county_col])
        flow_value = float(row[flow_col])

        # Flow direction is previous residence -> current residence.
        matrix[index_of[src_fips]][index_of[dst_fips]] = flow_value

        expected_dst = COUNTY_INFO[dst_fips]["sheet_name"]
        expected_src = COUNTY_INFO[src_fips]["sheet_name"]
        if row[current_name_col] != expected_dst or row[previous_name_col] != expected_src:
            raise ValueError(
                f"County name mismatch for flow row: {row[current_name_col]} / {row[previous_name_col]}"
            )

    return ordered_fips, matrix


def build_case_matrix(ordered_fips, weeks, county_weekly):
    case = []
    for week in weeks:
        row = []
        for fips in ordered_fips:
            row.append(county_weekly[fips].get(week, 0.0))
        case.append(row)
    return case


def build_graph_edges(matrix):
    edges = []
    flows = []
    for src_idx, row in enumerate(matrix):
        for dst_idx, value in enumerate(row):
            if src_idx == dst_idx or value <= 0:
                continue
            edges.append([src_idx, dst_idx])
            flows.append(float(value))
    return edges, flows


def write_region_csv(path, ordered_fips, coordinates):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "longitude", "latitude"])
        for fips in ordered_fips:
            longitude, latitude = coordinates[fips]
            writer.writerow([COUNTY_INFO[fips]["name"], longitude, latitude])


def build_payload(ordered_fips, weeks, population, case_matrix, matrix):
    edges, flows = build_graph_edges(matrix)
    return {
        "node": [COUNTY_INFO[fips]["name"] for fips in ordered_fips],
        "time": weeks,
        "population": [population[fips] for fips in ordered_fips],
        "case": case_matrix,
        "A": matrix,
        "G": edges,
        "flow": flows,
        "source_name": "New York State weekly laboratory-confirmed influenza cases by county",
        "source_url": "https://health.data.ny.gov/resource/jr8b-6gh6.csv",
        "movement_source_name": "U.S. Census county-to-county migration flows, 2011-2015 ACS",
        "movement_source_url": (
            "https://www2.census.gov/programs-surveys/demo/tables/geographic-mobility/2015/"
            "county-to-county-migration-2011-2015/county-to-county-migration-flows/"
            "county-to-county-2011-2015-current-residence-sort.xlsx"
        ),
        "population_source_url": (
            "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/"
            "co-est2024-alldata.csv"
        ),
        "notes": [
            "Exact county-level influenza counts; no county downscaling was used.",
            "Movement graph uses official county-to-county migration flow estimates within NYC.",
        ],
    }


def main():
    args = parse_args()

    county_weekly, weeks, coordinates = load_exact_flu_series(Path(args.flu_csv))
    weeks = filter_weeks(weeks, start_date=args.start_date, end_date=args.end_date)
    population = load_population(Path(args.population_csv))
    ordered_fips, matrix = load_exact_migration_matrix(Path(args.migration_xlsx))
    case_matrix = build_case_matrix(ordered_fips, weeks, county_weekly)
    payload = build_payload(ordered_fips, weeks, population, case_matrix, matrix)

    write_region_csv(Path(args.region_csv), ordered_fips, coordinates)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved regions CSV: {args.region_csv}")
    print(f"Nodes: {len(payload['node'])}")
    print(f"Weeks: {len(payload['time'])}")
    print(f"First week: {payload['time'][0]}")
    print(f"Last week: {payload['time'][-1]}")


if __name__ == "__main__":
    main()
