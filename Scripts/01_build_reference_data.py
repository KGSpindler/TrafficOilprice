from __future__ import annotations

import pandas as pd

from pipeline_utils import (
    ANALYSIS_END,
    ANALYSIS_START,
    CACHE_DIR,
    DAILY_RAIN_CACHE_FILE,
    HOLIDAY_CACHE_FILE,
    SITE_STATION_CACHE_FILE,
    STATIONS_CACHE_FILE,
    add_wgs84_coordinates,
    get_or_create_daily_rain,
    get_or_create_holiday_table,
    get_or_create_site_station_matches,
    get_or_create_stations,
    load_all_traffic,
    extract_valid_station_ids,
)

def main() -> None:
    print("Loading traffic data to identify road sites...")
    traffic = load_all_traffic()
    traffic = add_wgs84_coordinates(traffic)

    start_year = int(traffic["date"].dt.year.min())
    end_year = int(traffic["date"].dt.year.max())

    print("Building/loading holiday cache...")
    holidays_df = get_or_create_holiday_table(start_year, end_year)
    print(f"Saved/loaded holidays: {len(holidays_df)} rows -> {HOLIDAY_CACHE_FILE}")

    print("Building/loading DMI precipitation station cache...")
    stations = get_or_create_stations()
    print(f"Saved/loaded stations: {len(stations)} rows -> {STATIONS_CACHE_FILE}")

    print("Building/loading site-to-station matches...")
    matches = get_or_create_site_station_matches(traffic, stations)
    print(f"Saved/loaded matches: {len(matches)} rows -> {SITE_STATION_CACHE_FILE}")

    print("Building/loading daily rain cache only for matched stations...")
    station_ids = extract_valid_station_ids(matches["station_id"].tolist())
    rain = get_or_create_daily_rain(
        station_ids=station_ids,
        start_dt=pd.to_datetime(ANALYSIS_START),
        end_dt=pd.to_datetime(ANALYSIS_END),
    )
    print(f"Saved/loaded daily rain: {len(rain)} rows -> {DAILY_RAIN_CACHE_FILE}")

    print("\nStep 1 finished.")
    print(f"Cache folder: {CACHE_DIR}")

if __name__ == "__main__":
    main()
