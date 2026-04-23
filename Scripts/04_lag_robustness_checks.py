from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pipeline_utils import ANALYSIS_PANEL_FILE, OUTPUT_DIR


LAG_OUTPUT_DIR = Path(OUTPUT_DIR) / "lag_robustness"
LAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_clustered_ols(formula: str, data: pd.DataFrame):
    """
    Two-way clustered OLS when possible:
    - road/site
    - date
    Falls back to date clustering if needed.
    """
    road_codes = pd.factorize(data["road_site_id"].astype(str), sort=True)[0]
    date_codes = pd.factorize(pd.to_datetime(data["date"]).dt.normalize(), sort=True)[0]
    two_way_groups = np.column_stack([road_codes, date_codes])

    try:
        return smf.ols(formula=formula, data=data).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": two_way_groups,
                "use_correction": True,
                "df_correction": True,
            },
        )
    except Exception:
        return smf.ols(formula=formula, data=data).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": date_codes,
                "use_correction": True,
                "df_correction": True,
            },
        )



def save_summary(model, path: Path) -> None:
    cov_diag = np.diag(model.cov_params())
    non_positive_cov_diag = int(np.sum(cov_diag <= 0))

    with open(path, "w", encoding="utf-8") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in sqrt",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="covariance of constraints does not have full rank",
                category=UserWarning,
            )
            f.write(model.summary().as_text())

        if non_positive_cov_diag > 0:
            f.write(
                "\n\nNote: covariance matrix has non-positive diagonal entries for "
                f"{non_positive_cov_diag} coefficients; some coefficient-level SEs may be undefined."
            )



def add_time_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(float)
    return df



def add_price_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since fuel price is common across roads on a given day, create lags and moving
    averages from a unique date-level price series and merge back.
    """
    price_daily = (
        df[["date", "gasoline_price", "real_gasoline_price", "diesel_price"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .copy()
    )

    # Real gasoline lags
    price_daily["real_gas_lag1"] = price_daily["real_gasoline_price"].shift(1)
    price_daily["real_gas_lag3"] = price_daily["real_gasoline_price"].shift(3)
    price_daily["real_gas_lag7"] = price_daily["real_gasoline_price"].shift(7)
    price_daily["real_gas_lag14"] = price_daily["real_gasoline_price"].shift(14)
    price_daily["real_gas_lead7"] = price_daily["real_gasoline_price"].shift(-7)

    # Nominal gasoline lags
    price_daily["gas_lag7"] = price_daily["gasoline_price"].shift(7)

    # Moving averages: use only past/current information
    price_daily["real_gas_ma7"] = price_daily["real_gasoline_price"].rolling(window=7, min_periods=7).mean()
    price_daily["real_gas_ma14"] = price_daily["real_gasoline_price"].rolling(window=14, min_periods=14).mean()
    price_daily["real_gas_ma30"] = price_daily["real_gasoline_price"].rolling(window=30, min_periods=30).mean()
    price_daily["gas_ma7"] = price_daily["gasoline_price"].rolling(window=7, min_periods=7).mean()

    # Logs
    log_cols = {
        "real_gasoline_price": "log_real_gasoline_price",
        "real_gas_lag1": "log_real_gas_lag1",
        "real_gas_lag3": "log_real_gas_lag3",
        "real_gas_lag7": "log_real_gas_lag7",
        "real_gas_lag14": "log_real_gas_lag14",
        "real_gas_lead7": "log_real_gas_lead7",
        "real_gas_ma7": "log_real_gas_ma7",
        "real_gas_ma14": "log_real_gas_ma14",
        "real_gas_ma30": "log_real_gas_ma30",
        "gasoline_price": "log_gasoline_price",
        "gas_lag7": "log_gas_lag7",
        "gas_ma7": "log_gas_ma7",
        "diesel_price": "log_diesel_price",
    }
    for src, dst in log_cols.items():
        price_daily[dst] = np.where(price_daily[src] > 0, np.log(price_daily[src]), np.nan)

    keep_cols = ["date"] + list(log_cols.values())
    return df.merge(price_daily[keep_cols], on="date", how="left", suffixes=("", "_dup"))



def make_city_day_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to city-day because fuel price only varies by date.
    This is a useful robustness check.
    """
    agg = (
        df.groupby("date", as_index=False)
        .agg(
            traffic_total=("daily_traffic", "sum"),
            rain_mm_day=("rain_mm_day", "mean"),
            log_real_gasoline_price=("log_real_gasoline_price", "first"),
            log_real_gas_lag7=("log_real_gas_lag7", "first"),
            log_real_gas_ma7=("log_real_gas_ma7", "first"),
            log_real_gas_ma14=("log_real_gas_ma14", "first"),
            log_real_gas_ma30=("log_real_gas_ma30", "first"),
        )
        .sort_values("date")
    )
    agg["weekday"] = pd.to_datetime(agg["date"]).dt.dayofweek
    agg["month"] = pd.to_datetime(agg["date"]).dt.month
    agg["t"] = (pd.to_datetime(agg["date"]) - pd.to_datetime(agg["date"]).min()).dt.days.astype(float)
    agg = agg[agg["traffic_total"] > 0].copy()
    agg["log_traffic_total"] = np.log(agg["traffic_total"])
    return agg



def collect_result(model_name: str, coef_name: str, model, sample_desc: str) -> dict:
    return {
        "model": model_name,
        "coef_name": coef_name,
        "elasticity": model.params.get(coef_name, np.nan),
        "std_error": model.bse.get(coef_name, np.nan),
        "p_value": model.pvalues.get(coef_name, np.nan),
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared,
        "sample": sample_desc,
    }



def main() -> None:
    df = pd.read_csv(ANALYSIS_PANEL_FILE, parse_dates=["date"])
    if df.empty:
        raise ValueError("Analysis panel is empty. Run step 2 first.")

    df = add_time_trend(df)
    df = add_price_dynamics(df)

    results: list[dict] = []

    # Preferred base sample
    base = df.dropna(subset=["log_traffic", "log_real_gasoline_price", "rain_mm_day"]).copy()

    # 1. Main trend model
    m_main = fit_clustered_ols(
        "log_traffic ~ log_real_gasoline_price + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        base,
    )
    save_summary(m_main, LAG_OUTPUT_DIR / "model_real_gas_trend_main.txt")
    results.append(collect_result("real_gas_trend_main", "log_real_gasoline_price", m_main, "road-day"))

    # 2. One-week lag
    lag7 = df.dropna(subset=["log_traffic", "log_real_gas_lag7", "rain_mm_day"]).copy()
    m_lag7 = fit_clustered_ols(
        "log_traffic ~ log_real_gas_lag7 + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        lag7,
    )
    save_summary(m_lag7, LAG_OUTPUT_DIR / "model_real_gas_lag7_trend.txt")
    results.append(collect_result("real_gas_lag7_trend", "log_real_gas_lag7", m_lag7, "road-day"))

    # 3. One-day lag
    lag1 = df.dropna(subset=["log_traffic", "log_real_gas_lag1", "rain_mm_day"]).copy()
    m_lag1 = fit_clustered_ols(
        "log_traffic ~ log_real_gas_lag1 + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        lag1,
    )
    save_summary(m_lag1, LAG_OUTPUT_DIR / "model_real_gas_lag1_trend.txt")
    results.append(collect_result("real_gas_lag1_trend", "log_real_gas_lag1", m_lag1, "road-day"))

    # 4. 7-day moving average
    ma7 = df.dropna(subset=["log_traffic", "log_real_gas_ma7", "rain_mm_day"]).copy()
    m_ma7 = fit_clustered_ols(
        "log_traffic ~ log_real_gas_ma7 + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        ma7,
    )
    save_summary(m_ma7, LAG_OUTPUT_DIR / "model_real_gas_ma7_trend.txt")
    results.append(collect_result("real_gas_ma7_trend", "log_real_gas_ma7", m_ma7, "road-day"))

    # 5. 14-day moving average
    ma14 = df.dropna(subset=["log_traffic", "log_real_gas_ma14", "rain_mm_day"]).copy()
    m_ma14 = fit_clustered_ols(
        "log_traffic ~ log_real_gas_ma14 + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        ma14,
    )
    save_summary(m_ma14, LAG_OUTPUT_DIR / "model_real_gas_ma14_trend.txt")
    results.append(collect_result("real_gas_ma14_trend", "log_real_gas_ma14", m_ma14, "road-day"))

    # 6. 30-day moving average
    ma30 = df.dropna(subset=["log_traffic", "log_real_gas_ma30", "rain_mm_day"]).copy()
    m_ma30 = fit_clustered_ols(
        "log_traffic ~ log_real_gas_ma30 + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        ma30,
    )
    save_summary(m_ma30, LAG_OUTPUT_DIR / "model_real_gas_ma30_trend.txt")
    results.append(collect_result("real_gas_ma30_trend", "log_real_gas_ma30", m_ma30, "road-day"))

    # 7. Distributed lag lite: current + lag7
    dl = df.dropna(subset=["log_traffic", "log_real_gasoline_price", "log_real_gas_lag7", "rain_mm_day"]).copy()
    m_dl = fit_clustered_ols(
        "log_traffic ~ log_real_gasoline_price + log_real_gas_lag7 + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        dl,
    )
    save_summary(m_dl, LAG_OUTPUT_DIR / "model_real_gas_current_and_lag7_trend.txt")
    results.append(collect_result("real_gas_current_lag7_trend", "log_real_gasoline_price", m_dl, "road-day"))
    results.append(collect_result("real_gas_current_lag7_trend", "log_real_gas_lag7", m_dl, "road-day"))
    results.append(
        {
            "model": "real_gas_current_lag7_trend",
            "coef_name": "sum_current_plus_lag7",
            "elasticity": m_dl.params.get("log_real_gasoline_price", np.nan) + m_dl.params.get("log_real_gas_lag7", np.nan),
            "std_error": np.nan,
            "p_value": np.nan,
            "n_obs": int(m_dl.nobs),
            "r_squared": m_dl.rsquared,
            "sample": "road-day",
        }
    )

    # 8. Month-FE version with MA7 as sensitivity check
    m_month_ma7 = fit_clustered_ols(
        "log_traffic ~ log_real_gas_ma7 + rain_mm_day + C(weekday) + C(month) + C(road_site_id)",
        ma7,
    )
    save_summary(m_month_ma7, LAG_OUTPUT_DIR / "model_real_gas_ma7_fe_month.txt")
    results.append(collect_result("real_gas_ma7_fe_month", "log_real_gas_ma7", m_month_ma7, "road-day"))

    # 9. Placebo lead: should ideally be weak/zero
    lead7 = df.dropna(subset=["log_traffic", "log_real_gas_lead7", "rain_mm_day"]).copy()
    m_lead7 = fit_clustered_ols(
        "log_traffic ~ log_real_gas_lead7 + rain_mm_day + C(weekday) + C(road_site_id) + t + I(t**2)",
        lead7,
    )
    save_summary(m_lead7, LAG_OUTPUT_DIR / "model_real_gas_lead7_placebo.txt")
    results.append(collect_result("real_gas_lead7_placebo", "log_real_gas_lead7", m_lead7, "road-day"))

    # 10. City-day aggregate robustness
    city = make_city_day_panel(df)
    city_main = city.dropna(subset=["log_traffic_total", "log_real_gasoline_price", "rain_mm_day"]).copy()
    m_city_main = smf.ols(
        "log_traffic_total ~ log_real_gasoline_price + rain_mm_day + C(weekday) + t + I(t**2)",
        data=city_main,
    ).fit(cov_type="HC1")
    save_summary(m_city_main, LAG_OUTPUT_DIR / "model_city_day_real_gas_trend.txt")
    results.append(collect_result("city_day_real_gas_trend", "log_real_gasoline_price", m_city_main, "city-day"))

    city_ma7 = city.dropna(subset=["log_traffic_total", "log_real_gas_ma7", "rain_mm_day"]).copy()
    m_city_ma7 = smf.ols(
        "log_traffic_total ~ log_real_gas_ma7 + rain_mm_day + C(weekday) + t + I(t**2)",
        data=city_ma7,
    ).fit(cov_type="HC1")
    save_summary(m_city_ma7, LAG_OUTPUT_DIR / "model_city_day_real_gas_ma7_trend.txt")
    results.append(collect_result("city_day_real_gas_ma7_trend", "log_real_gas_ma7", m_city_ma7, "city-day"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(LAG_OUTPUT_DIR / "lag_robustness_summary.csv", index=False)

    display_cols = ["model", "coef_name", "elasticity", "std_error", "p_value", "n_obs", "r_squared", "sample"]
    print(results_df[display_cols].to_string(index=False))
    print(f"\nSaved lag and robustness outputs to {LAG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
