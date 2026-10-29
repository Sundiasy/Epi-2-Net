import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
BASE_JSON = PROJECT_ROOT / "dataset" / "California" / "COVIDinCalifornia.json"
SOURCE_CSV = THIS_DIR / "flu-publichealthlab-byregion-fluseason.csv"
OUTPUT_JSON = THIS_DIR / "InfluenzaLabInCalifornia.json"


# CDPH influenza surveillance uses six California regions.
# Source for county grouping:
# - CDPH Regional Public Health Office pages for Bay Area / Central / Rural North /
#   Greater Sierra Sacramento / Southern California / Los Angeles regions.
# - "Upper Southern" county definition is also stated in a 2019 KPBS article that
#   cites CDPH influenza reporting regions: Kern, Los Angeles, San Luis Obispo,
#   Santa Barbara, Ventura. The remaining southern counties are assigned to
#   "Lower Southern".
REGION_TO_COUNTIES = {
    "Bay Area": [
        "Alameda",
        "Contra Costa",
        "Marin",
        "Monterey",
        "Napa",
        "San Benito",
        "San Francisco",
        "San Mateo",
        "Santa Clara",
        "Santa Cruz",
        "Solano",
        "Sonoma",
    ],
    "Central": [
        "Calaveras",
        "Fresno",
        "Kings",
        "Madera",
        "Mariposa",
        "Merced",
        "San Joaquin",
        "Stanislaus",
        "Tulare",
        "Tuolumne",
    ],
    "Northern": [
        "Del Norte",
        "Glenn",
        "Humboldt",
        "Lake",
        "Lassen",
        "Mendocino",
        "Modoc",
        "Shasta",
        "Siskiyou",
        "Tehama",
        "Trinity",
    ],
    "Sacramento": [
        "Alpine",
        "Amador",
        "Butte",
        "Colusa",
        "El Dorado",
        "Nevada",
        "Placer",
        "Plumas",
        "Sacramento",
        "Sierra",
        "Sutter",
        "Yolo",
        "Yuba",
    ],
    "Upper Southern": [
        "Kern",
        "Los Angeles",
        "San Luis Obispo",
        "Santa Barbara",
        "Ventura",
    ],
    "Lower Southern": [
        "Imperial",
        "Inyo",
        "Mono",
        "Orange",
        "Riverside",
        "San Bernardino",
        "San Diego",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a county-level California influenza dataset for Epi-2-Net."
    )
    parser.add_argument("--base-json", default=str(BASE_JSON))
    parser.add_argument("--source-csv", default=str(SOURCE_CSV))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    return parser.parse_args()


def parse_weekending(text):
    text = text.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {text}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_county_lookup(base_payload):
    counties = base_payload["node"]
    populations = base_payload["population"]
    if len(counties) != len(populations):
        raise ValueError("Base payload `node` and `population` lengths do not match.")
    return counties, {county: float(pop) for county, pop in zip(counties, populations)}


def validate_region_mapping(counties):
    county_set = set(counties)
    seen = set()
    for region, region_counties in REGION_TO_COUNTIES.items():
        for county in region_counties:
            if county not in county_set:
                raise ValueError(f"County `{county}` from region `{region}` is missing in base payload.")
            seen.add(county)

    missing = sorted(county_set - seen)
    if missing:
        raise ValueError(f"Counties missing from region mapping: {missing}")


def build_county_to_region():
    mapping = {}
    for region, counties in REGION_TO_COUNTIES.items():
        for county in counties:
            mapping[county] = region
    return mapping


def load_region_weekly_cases(csv_path):
    region_weekly = defaultdict(lambda: defaultdict(float))
    week_dates = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row["region"].strip()
            category = row["Influenza_Category"].strip()
            if region == "California" or category != "Influenza_Total":
                continue

            week = parse_weekending(row["weekending"])
            count = float(row["Count"])
            region_weekly[region][week.isoformat()] += count
            week_dates.add(week.isoformat())

    if not region_weekly:
        raise ValueError("No regional influenza counts were loaded from the source CSV.")

    return region_weekly, sorted(week_dates)


def build_region_population_weights(county_populations):
    weights = {}
    for region, counties in REGION_TO_COUNTIES.items():
        region_population = sum(county_populations[county] for county in counties)
        if region_population <= 0:
            raise ValueError(f"Region `{region}` has non-positive population.")
        weights[region] = {
            county: county_populations[county] / region_population for county in counties
        }
    return weights


def build_case_matrix(counties, week_dates, region_weekly, weights, county_to_region):
    case_matrix = []
    for week in week_dates:
        row = []
        for county in counties:
            region = county_to_region[county]
            region_total = region_weekly.get(region, {}).get(week, 0.0)
            row.append(region_total * weights[region][county])
        case_matrix.append(row)
    return case_matrix


def build_output_payload(base_payload, week_dates, case_matrix):
    output = {
        "node": base_payload["node"],
        "time": week_dates,
        "population": base_payload["population"],
        "case": case_matrix,
        "A": base_payload.get("A", []),
        "G": base_payload.get("G", []),
        "flow": base_payload.get("flow", []),
        "source_name": "California public health laboratory influenza surveillance",
        "source_url": (
            "https://data.chhs.ca.gov/dataset/fc544658-35c5-4be0-af20-fc703bc57c13/"
            "resource/063faf93-0572-4fdc-84cf-263f110dc09a/download/"
            "flu-publichealthlab-byregion-fluseason.csv"
        ),
        "source_category": "Influenza_Total",
        "notes": [
            "Regional influenza totals are distributed to counties by population share.",
            "County population and mobility graph are inherited from dataset/California/COVIDinCalifornia.json.",
            "Berkeley is merged into Alameda County; Long Beach and Pasadena are merged into Los Angeles County.",
        ],
    }
    return output


def main():
    args = parse_args()
    base_json = Path(args.base_json)
    source_csv = Path(args.source_csv)
    output_json = Path(args.output_json)

    base_payload = load_json(base_json)
    counties, county_populations = build_county_lookup(base_payload)
    validate_region_mapping(counties)
    county_to_region = build_county_to_region()
    region_weekly, week_dates = load_region_weekly_cases(source_csv)
    weights = build_region_population_weights(county_populations)
    case_matrix = build_case_matrix(counties, week_dates, region_weekly, weights, county_to_region)
    output_payload = build_output_payload(base_payload, week_dates, case_matrix)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False)

    print(f"Saved influenza dataset to: {output_json}")
    print(f"Nodes: {len(output_payload['node'])}")
    print(f"Weeks: {len(output_payload['time'])}")
    print(f"First week: {output_payload['time'][0]}")
    print(f"Last week: {output_payload['time'][-1]}")


if __name__ == "__main__":
    main()
