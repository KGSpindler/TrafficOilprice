from __future__ import annotations

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figure_data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Inputs expected from your existing pipeline
ANALYSIS_PANEL_FILE = OUTPUT_DIR / "analysis_panel_daily.csv"
RESULTS_SUMMARY_FILE = OUTPUT_DIR / "model_results_summary.csv"
SITE_MATCHES_FILE = OUTPUT_DIR / "site_station_matches_used.csv"

# Optional lag/robustness files from step 4
LAG_RESULTS_SUMMARY_FILE = OUTPUT_DIR / "lag_robustness" / "lag_robustness_summary.csv"


# -----------------------------
# Helpers
# -----------------------------

def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return None


def ensure_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def maybe_add_month_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    out["month"] = pd.to_datetime(out["date"]).dt.month
    out["year_month"] = pd.to_datetime(out["date"]).dt.to_period("M").astype(str)
    return out


def tidy_model_label(model_name: str) -> str:
    mapping = {
        "real_gasoline_trend_main": "Real gasoline, trend",
        "real_gasoline_fe_month": "Real gasoline, month FE",
        "nominal_gasoline_trend": "Nominal gasoline, trend",
        "nominal_diesel_trend": "Nominal diesel, trend",
        "real_gasoline_daily": "Real gasoline, daily",
        "nominal_gasoline_daily": "Nominal gasoline, daily",
        "nominal_diesel_daily": "Nominal diesel, daily",
        "real_gasoline_road_weekday_fe": "Real gasoline, road×weekday FE",
        "real_gas_lag1_trend": "Real gasoline, lag 1",
        "real_gas_lag7_trend": "Real gasoline, lag 7",
        "real_gas_ma7_trend": "Real gasoline, MA7",
        "real_gas_ma14_trend": "Real gasoline, MA14",
        "real_gas_ma30_trend": "Real gasoline, MA30",
        "real_gas_ma7_fe_month": "Real gasoline, MA7 month FE",
        "real_gas_current_and_lag7_trend": "Real gasoline, current + lag 7",
        "real_gas_lead7_placebo": "Real gasoline, lead 7 placebo",
        "city_day_real_gas_trend": "City-day real gasoline",
        "city_day_real_gas_ma7_trend": "City-day real gasoline, MA7",
    }
    return mapping.get(model_name, model_name.replace("_", " ").title())


def infer_model_group(model_name: str) -> str:
    if "city_day" in model_name:
        return "city_day"
    if "lead" in model_name:
        return "placebo"
    if "lag" in model_name:
        return "lag"
    if "ma" in model_name:
        return "moving_average"
    if "diesel" in model_name:
        return "diesel"
    if "month" in model_name:
        return "month_fe"
    return "main"


# -----------------------------
# Build coefficient datasets
# -----------------------------

def build_coefficient_datasets() -> None:
    main = read_csv_if_exists(RESULTS_SUMMARY_FILE)
    lag = read_csv_if_exists(LAG_RESULTS_SUMMARY_FILE)

    frames = []
    if main is not None:
        frames.append(main)
    if lag is not None:
        frames.append(lag)

    if not frames:
        print("No model summary files found; skipping coefficient datasets.")
        return

    df = pd.concat(frames, ignore_index=True)
    ensure_columns(df, ["model", "elasticity", "std_error", "p_value"], "model results summary")

    df = df.copy()
    df["label"] = df["model"].map(tidy_model_label)
    df["group"] = df["model"].map(infer_model_group)
    df["ci_low_95"] = df["elasticity"] - 1.96 * df["std_error"]
    df["ci_high_95"] = df["elasticity"] + 1.96 * df["std_error"]
    df["significant_5pct"] = df["p_value"] < 0.05

    # Order for coefficient plot
    order = [
        "real_gasoline_trend_main",
        "real_gas_lag1_trend",
        "real_gas_lag7_trend",
        "real_gas_ma7_trend",
        "real_gas_ma14_trend",
        "real_gas_ma30_trend",
        "real_gas_current_and_lag7_trend",
        "real_gas_lead7_placebo",
        "city_day_real_gas_trend",
        "city_day_real_gas_ma7_trend",
        "real_gasoline_fe_month",
        "real_gas_ma7_fe_month",
        "nominal_gasoline_trend",
        "nominal_diesel_trend",
    ]
    order_map = {name: i for i, name in enumerate(order)}
    df["plot_order"] = df["model"].map(lambda x: order_map.get(x, 999))
    df = df.sort_values(["plot_order", "label"]).reset_index(drop=True)

    df.to_csv(FIGURE_DIR / "coefficients_all_models.csv", index=False)

    lag_only = df[df["group"].isin(["main", "lag", "moving_average", "placebo", "city_day"])].copy()
    lag_only.to_csv(FIGURE_DIR / "coefficients_lag_robustness.csv", index=False)

    panel_vs_city = df[df["model"].isin([
        "real_gasoline_trend_main",
        "real_gas_lag7_trend",
        "real_gas_ma7_trend",
        "city_day_real_gas_trend",
        "city_day_real_gas_ma7_trend",
    ])].copy()
    panel_vs_city.to_csv(FIGURE_DIR / "coefficients_panel_vs_city.csv", index=False)


# -----------------------------
# Build descriptive/base-data datasets
# -----------------------------

def build_descriptive_datasets() -> None:
    panel = read_csv_if_exists(ANALYSIS_PANEL_FILE, parse_dates=["date"])
    if panel is None:
        raise FileNotFoundError(f"Missing analysis panel: {ANALYSIS_PANEL_FILE}")

    required = ["date", "road_site_id", "daily_traffic", "rain_mm_day"]
    ensure_columns(panel, required, "analysis panel")

    panel = maybe_add_month_keys(panel)
    panel["weekday_name"] = pd.to_datetime(panel["date"]).dt.day_name()

    # Daily city totals / averages for time series figures
    agg_dict = {
        "daily_traffic": ["sum", "mean", "median"],
        "rain_mm_day": ["mean", "median", "max"],
    }
    optional_cols = [
        "gasoline_price",
        "real_gasoline_price",
        "diesel_price",
        "real_diesel_price",
    ]
    for col in optional_cols:
        if col in panel.columns:
            agg_dict[col] = "mean"

    city_day = panel.groupby("date").agg(agg_dict)
    city_day.columns = ["_".join(c).strip("_") for c in city_day.columns.to_flat_index()]
    city_day = city_day.reset_index().sort_values("date")

    # Rename cleaner
    city_day = city_day.rename(columns={
        "daily_traffic_sum": "traffic_total",
        "daily_traffic_mean": "traffic_mean_site",
        "daily_traffic_median": "traffic_median_site",
        "rain_mm_day_mean": "rain_mean_mm",
        "rain_mm_day_median": "rain_median_mm",
        "rain_mm_day_max": "rain_max_mm",
    })

    # Indexed versions for overlay figures
    for col in ["traffic_total", "traffic_mean_site", "gasoline_price_mean", "real_gasoline_price_mean", "diesel_price_mean", "real_diesel_price_mean"]:
        if col in city_day.columns:
            first_val = city_day[col].dropna().iloc[0]
            if first_val and np.isfinite(first_val):
                city_day[f"{col}_index100"] = city_day[col] / first_val * 100

    # Rolling smoothed series for web plots
    for col in ["traffic_total", "traffic_mean_site", "gasoline_price_mean", "real_gasoline_price_mean", "rain_mean_mm"]:
        if col in city_day.columns:
            city_day[f"{col}_ma7"] = city_day[col].rolling(7, min_periods=1).mean()
            city_day[f"{col}_ma30"] = city_day[col].rolling(30, min_periods=1).mean()

    city_day.to_csv(FIGURE_DIR / "city_day_timeseries.csv", index=False)

    # Weekly aggregate for smoother plots
    city_week = city_day.copy()
    city_week["week_start"] = pd.to_datetime(city_week["date"]).dt.to_period("W-MON").apply(lambda p: p.start_time)
    value_cols = [c for c in city_week.columns if c not in {"date", "week_start"}]
    city_week = city_week.groupby("week_start", as_index=False)[value_cols].mean(numeric_only=True)
    city_week.to_csv(FIGURE_DIR / "city_week_timeseries.csv", index=False)

    # Monthly aggregate for prices / traffic / rain overview
    city_month = panel.groupby("year_month", as_index=False).agg(
        traffic_total=("daily_traffic", "sum"),
        traffic_mean_site=("daily_traffic", "mean"),
        rain_mean_mm=("rain_mm_day", "mean"),
        rain_total_mm=("rain_mm_day", "sum"),
    )
    for col in ["gasoline_price", "real_gasoline_price", "diesel_price", "real_diesel_price"]:
        if col in panel.columns:
            city_month[col] = panel.groupby("year_month")[col].mean().values
    city_month.to_csv(FIGURE_DIR / "city_month_timeseries.csv", index=False)

    # Weekday profile
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_profile = panel.groupby("weekday_name", as_index=False).agg(
        traffic_mean=("daily_traffic", "mean"),
        traffic_median=("daily_traffic", "median"),
        rain_mean_mm=("rain_mm_day", "mean"),
        n_obs=("daily_traffic", "size"),
    )
    weekday_profile["weekday_order"] = weekday_profile["weekday_name"].map({d: i for i, d in enumerate(weekday_order)})
    weekday_profile = weekday_profile.sort_values("weekday_order")
    weekday_profile.to_csv(FIGURE_DIR / "weekday_profile.csv", index=False)

    # Monthly seasonality profile
    month_profile = panel.groupby("month", as_index=False).agg(
        traffic_mean=("daily_traffic", "mean"),
        traffic_median=("daily_traffic", "median"),
        rain_mean_mm=("rain_mm_day", "mean"),
        n_obs=("daily_traffic", "size"),
    )
    month_profile.to_csv(FIGURE_DIR / "month_profile.csv", index=False)

    # Rain distribution / bins
    rain_bins = [-0.001, 0, 2, 5, 10, np.inf]
    rain_labels = ["0 mm", "0-2 mm", "2-5 mm", "5-10 mm", "10+ mm"]
    rain_df = panel[["date", "daily_traffic", "rain_mm_day"]].copy()
    rain_df["rain_bin"] = pd.cut(rain_df["rain_mm_day"], bins=rain_bins, labels=rain_labels)
    rain_bin_stats = rain_df.groupby("rain_bin", observed=False, as_index=False).agg(
        traffic_mean=("daily_traffic", "mean"),
        traffic_median=("daily_traffic", "median"),
        rain_mean_mm=("rain_mm_day", "mean"),
        n_obs=("daily_traffic", "size"),
    )
    rain_bin_stats.to_csv(FIGURE_DIR / "rain_bin_traffic.csv", index=False)

    rain_hist = panel["rain_mm_day"].value_counts(bins=40, sort=False).reset_index()
    rain_hist.columns = ["rain_bin", "count"]
    rain_hist["bin_left"] = rain_hist["rain_bin"].apply(lambda x: x.left)
    rain_hist["bin_right"] = rain_hist["rain_bin"].apply(lambda x: x.right)
    rain_hist = rain_hist.drop(columns=["rain_bin"])
    rain_hist.to_csv(FIGURE_DIR / "rain_histogram_bins.csv", index=False)

    # Traffic distribution across sites
    site_summary = panel.groupby("road_site_id", as_index=False).agg(
        n_days=("daily_traffic", "size"),
        avg_daily_traffic=("daily_traffic", "mean"),
        median_daily_traffic=("daily_traffic", "median"),
        std_daily_traffic=("daily_traffic", "std"),
        rain_mean_mm=("rain_mm_day", "mean"),
    )
    for col in ["road_name", "lane", "x_utm32", "y_utm32", "lon", "lat", "station_id", "station_distance_km"]:
        if col in panel.columns:
            site_summary[col] = panel.groupby("road_site_id")[col].first().values
    site_summary.to_csv(FIGURE_DIR / "site_summary.csv", index=False)

    # Station map / map-ready site points
    site_points = panel.groupby("road_site_id", as_index=False).agg(
        avg_daily_traffic=("daily_traffic", "mean"),
        median_daily_traffic=("daily_traffic", "median"),
        n_days=("daily_traffic", "size"),
        rain_mean_mm=("rain_mm_day", "mean"),
    )
    for col in ["road_name", "lane", "x_utm32", "y_utm32", "lon", "lat", "station_id", "station_distance_km"]:
        if col in panel.columns:
            site_points[col] = panel.groupby("road_site_id")[col].first().values
    site_points.to_csv(FIGURE_DIR / "site_points_map.csv", index=False)

    matches = read_csv_if_exists(SITE_MATCHES_FILE)
    if matches is not None:
        matches.to_csv(FIGURE_DIR / "site_station_matches.csv", index=False)

    # Base data table for website cards/summary metrics
    summary_cards = {
        "n_observations": int(len(panel)),
        "n_sites": int(panel["road_site_id"].nunique()),
        "start_date": str(panel["date"].min().date()),
        "end_date": str(panel["date"].max().date()),
        "mean_daily_traffic": float(panel["daily_traffic"].mean()),
        "median_daily_traffic": float(panel["daily_traffic"].median()),
        "mean_rain_mm": float(panel["rain_mm_day"].mean()),
        "share_rainy_days": float((panel["rain_mm_day"] > 0).mean()),
    }
    for col in ["gasoline_price", "real_gasoline_price", "diesel_price", "real_diesel_price"]:
        if col in panel.columns:
            summary_cards[f"mean_{col}"] = float(panel[col].mean())
    with open(FIGURE_DIR / "summary_cards.json", "w", encoding="utf-8") as f:
        json.dump(summary_cards, f, ensure_ascii=False, indent=2)


# -----------------------------
# Build model-specific datasets
# -----------------------------

def build_model_specific_datasets() -> None:
    # Website-friendly explanation file for labels/definitions
    definitions = [
        {
            "key": "MA7",
            "label": "7-day moving average",
            "description": "The average value over the current day and the previous 6 days. Used to smooth short-run noise and capture delayed responses.",
        },
        {
            "key": "MA14",
            "label": "14-day moving average",
            "description": "The average value over the current day and the previous 13 days.",
        },
        {
            "key": "MA30",
            "label": "30-day moving average",
            "description": "The average value over the current day and the previous 29 days.",
        },
        {
            "key": "lag1",
            "label": "1-day lag",
            "description": "Yesterday's value.",
        },
        {
            "key": "lag7",
            "label": "7-day lag",
            "description": "The value from 7 days earlier.",
        },
        {
            "key": "lead7",
            "label": "7-day lead/placebo",
            "description": "The value 7 days in the future; used as a placebo check.",
        },
    ]
    pd.DataFrame(definitions).to_csv(FIGURE_DIR / "model_term_definitions.csv", index=False)


def main() -> None:
    build_coefficient_datasets()
    build_descriptive_datasets()
    build_model_specific_datasets()
    print(f"Saved figure-ready datasets to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
