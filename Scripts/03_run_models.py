from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pipeline_utils import ANALYSIS_PANEL_FILE, OUTPUT_DIR


def fit_clustered_ols(formula: str, data: pd.DataFrame):
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


def save_summary(model, path):
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


def resolve_analysis_panel_file() -> tuple[pd.DataFrame, str]:
    candidates = [
        ANALYSIS_PANEL_FILE,
        OUTPUT_DIR / "analysis_panel.csv",
    ]

    for panel_path in candidates:
        if panel_path.exists():
            return pd.read_csv(panel_path, parse_dates=["date"]), str(panel_path)

    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find an analysis panel file for modeling. "
        "Run Scripts/02_prepare_analysis_panel.py first.\n"
        f"Checked paths:\n{candidate_text}"
    )


def main() -> None:
    df, panel_path = resolve_analysis_panel_file()
    if df.empty:
        raise ValueError("Analysis panel is empty. Run step 2 first.")

    df = add_time_trend(df)

    # Main preferred model:
    # same road compared to itself, same weekday structure, local rain, plus smooth time trend
    real_gas_df = df.dropna(subset=["log_traffic", "log_real_gasoline_price", "rain_mm_day"]).copy()
    model_real_gas_trend = fit_clustered_ols(
        formula=(
            "log_traffic ~ log_real_gasoline_price + rain_mm_day "
            "+ C(weekday) + C(road_site_id) + t + I(t**2)"
        ),
        data=real_gas_df,
    )

    # Simpler FE model without trend, for comparison
    model_real_gas_fe = fit_clustered_ols(
        formula=(
            "log_traffic ~ log_real_gasoline_price + rain_mm_day "
            "+ C(weekday) + C(month) + C(road_site_id)"
        ),
        data=real_gas_df,
    )

    # Nominal gasoline robustness
    nominal_gas_df = df.dropna(subset=["log_traffic", "log_gasoline_price", "rain_mm_day"]).copy()
    model_nominal_gas_trend = fit_clustered_ols(
        formula=(
            "log_traffic ~ log_gasoline_price + rain_mm_day "
            "+ C(weekday) + C(road_site_id) + t + I(t**2)"
        ),
        data=nominal_gas_df,
    )

    # Nominal diesel robustness
    diesel_df = df.dropna(subset=["log_traffic", "log_diesel_price", "rain_mm_day"]).copy()
    model_nominal_diesel_trend = fit_clustered_ols(
        formula=(
            "log_traffic ~ log_diesel_price + rain_mm_day "
            "+ C(weekday) + C(road_site_id) + t + I(t**2)"
        ),
        data=diesel_df,
    )

    save_summary(model_real_gas_trend, OUTPUT_DIR / "model_real_gasoline_trend.txt")
    save_summary(model_real_gas_fe, OUTPUT_DIR / "model_real_gasoline_fe.txt")
    save_summary(model_nominal_gas_trend, OUTPUT_DIR / "model_nominal_gasoline_trend.txt")
    save_summary(model_nominal_diesel_trend, OUTPUT_DIR / "model_nominal_diesel_trend.txt")

    results = pd.DataFrame(
        [
            {
                "model": "real_gasoline_trend_main",
                "coef_name": "log_real_gasoline_price",
                "elasticity": model_real_gas_trend.params.get("log_real_gasoline_price", np.nan),
                "std_error": model_real_gas_trend.bse.get("log_real_gasoline_price", np.nan),
                "p_value": model_real_gas_trend.pvalues.get("log_real_gasoline_price", np.nan),
                "n_obs": int(model_real_gas_trend.nobs),
                "r_squared": model_real_gas_trend.rsquared,
            },
            {
                "model": "real_gasoline_fe_month",
                "coef_name": "log_real_gasoline_price",
                "elasticity": model_real_gas_fe.params.get("log_real_gasoline_price", np.nan),
                "std_error": model_real_gas_fe.bse.get("log_real_gasoline_price", np.nan),
                "p_value": model_real_gas_fe.pvalues.get("log_real_gasoline_price", np.nan),
                "n_obs": int(model_real_gas_fe.nobs),
                "r_squared": model_real_gas_fe.rsquared,
            },
            {
                "model": "nominal_gasoline_trend",
                "coef_name": "log_gasoline_price",
                "elasticity": model_nominal_gas_trend.params.get("log_gasoline_price", np.nan),
                "std_error": model_nominal_gas_trend.bse.get("log_gasoline_price", np.nan),
                "p_value": model_nominal_gas_trend.pvalues.get("log_gasoline_price", np.nan),
                "n_obs": int(model_nominal_gas_trend.nobs),
                "r_squared": model_nominal_gas_trend.rsquared,
            },
            {
                "model": "nominal_diesel_trend",
                "coef_name": "log_diesel_price",
                "elasticity": model_nominal_diesel_trend.params.get("log_diesel_price", np.nan),
                "std_error": model_nominal_diesel_trend.bse.get("log_diesel_price", np.nan),
                "p_value": model_nominal_diesel_trend.pvalues.get("log_diesel_price", np.nan),
                "n_obs": int(model_nominal_diesel_trend.nobs),
                "r_squared": model_nominal_diesel_trend.rsquared,
            },
        ]
    )

    results.to_csv(OUTPUT_DIR / "model_results_summary.csv", index=False)

    print(f"Loaded analysis panel from {panel_path}")
    print(results.to_string(index=False))
    print(f"\nSaved model summaries and results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()