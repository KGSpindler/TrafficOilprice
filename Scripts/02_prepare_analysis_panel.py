from __future__ import annotations

import pandas as pd

from pipeline_utils import (
    ANALYSIS_PANEL_FILE,
    OUTPUT_DIR,
    add_calendar_flags,
    add_real_price_columns,
    add_wgs84_coordinates,
    drop_holidays_and_preholidays,
    get_or_create_daily_rain,
    get_or_create_holiday_table,
    get_or_create_site_station_matches,
    get_or_create_stations,
    load_all_traffic,
    load_fuel_data,
    parse_cpi_file,
    CPI_FILE,
    extract_valid_station_ids,
)

def main() -> None:
    print("Loading traffic...")
    traffic = load_all_traffic()

    print("Adding calendar flags and dropping holidays/pre-holidays...")
    hol_df = get_or_create_holiday_table(
        int(traffic["date"].dt.year.min()),
        int(traffic["date"].dt.year.max()),
    )
    traffic = add_calendar_flags(traffic, hol_df=hol_df)
    traffic = drop_holidays_and_preholidays(traffic)
    traffic = add_wgs84_coordinates(traffic)

    print("Loading station matches and daily rain...")
    stations = get_or_create_stations()
    site_station = get_or_create_site_station_matches(traffic, stations)
    traffic = traffic.merge(site_station, on="road_site_id", how="left")
    traffic = traffic[traffic["station_id"].notna()].copy()
    traffic["station_id"] = traffic["station_id"].astype(str)

    station_ids = extract_valid_station_ids(traffic["station_id"].tolist())
    rain = get_or_create_daily_rain(
        station_ids=station_ids,
        start_dt=traffic["date"].min(),
        end_dt=traffic["date"].max(),
    )
    rain = rain.reindex(columns=["station_id", "date", "rain_mm_day", "rained_day"])

    print("Loading fuel prices and CPI...")
    fuel = load_fuel_data()
    cpi = parse_cpi_file(CPI_FILE)

    print("Merging everything...")
    df = traffic.merge(
        rain[["station_id", "date", "rain_mm_day", "rained_day"]],
        on=["station_id", "date"],
        how="left",
    )
    df = df.merge(
        fuel[["date", "gasoline_price", "diesel_price"]],
        on="date",
        how="left",
    )
    df = df.merge(
        cpi[["year", "month", "cpi_total", "cpi_liquid_fuels"]],
        on=["year", "month"],
        how="left",
    )

    df["rain_mm_day"] = df["rain_mm_day"].fillna(0.0)
    df["rained_day"] = df["rained_day"].fillna(0).astype(int)

    df = add_real_price_columns(df)

    # Keep only rows with traffic and at least nominal gasoline price.
    df = df[df["daily_traffic"] > 0].copy()
    df = df[df["gasoline_price"].notna()].copy()

    # Reorder some core columns for readability.
    front_cols = [
        "date", "road_site_id", "road_name", "station_id", "station_distance_km",
        "daily_traffic", "rain_mm_day", "rained_day",
        "gasoline_price", "diesel_price",
        "cpi_total", "cpi_liquid_fuels",
        "real_gasoline_price", "real_diesel_price",
        "weekday", "weekday_name", "month", "year",
    ]
    existing_front_cols = [c for c in front_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_front_cols]
    df = df[existing_front_cols + remaining_cols]

    df.to_csv(ANALYSIS_PANEL_FILE, index=False)

    matches_used = (
        df[["road_site_id", "road_name", "station_id", "station_distance_km"]]
        .drop_duplicates()
        .sort_values(["road_name", "road_site_id"])
    )
    matches_used.to_csv(OUTPUT_DIR / "site_station_matches_used.csv", index=False)

    cpi.to_csv(OUTPUT_DIR / "cpi_monthly_long.csv", index=False)

    print(f"Saved merged analysis panel -> {ANALYSIS_PANEL_FILE}")
    print(f"Saved CPI long file -> {OUTPUT_DIR / 'cpi_monthly_long.csv'}")
    print(f"Rows in final panel: {len(df)}")

if __name__ == "__main__":
    main()
