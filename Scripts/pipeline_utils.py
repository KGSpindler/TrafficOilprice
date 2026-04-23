from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import holidays
import numpy as np
import pandas as pd
import requests
from pandas.errors import EmptyDataError
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "BaseDatasets"
TRAFFIC_DATA_DIR = DATA_DIR / "Traffic 2010-2024"
FUEL_DATA_DIR = DATA_DIR / "Oil Prices 2010-2024"

TRAFFIC_FILES = [
    TRAFFIC_DATA_DIR / "faste-trafiktaellinger-2011.xlsx",
    TRAFFIC_DATA_DIR / "faste-trafiktaellinger-2012.xlsx",
    TRAFFIC_DATA_DIR / "faste-trafiktaellinger-2013.xlsx",
    TRAFFIC_DATA_DIR / "faste-trafiktaellinger-2014.xlsx",
]

FUEL_GAS_FILE = FUEL_DATA_DIR / "ok_prisudvikling_blyfri-95-(e10)_2010-2014_inkl.-moms_inkl.-afgifter_pr.-liter.csv"
FUEL_DIESEL_FILE = FUEL_DATA_DIR / "ok_prisudvikling_diesel_2010-2014_inkl.-moms_inkl.-afgifter_pr.-liter.csv"
CPI_FILE = FUEL_DATA_DIR / "CPI Total and Liquid Fuels.csv"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

HOLIDAY_CACHE_FILE = CACHE_DIR / "holiday_calendar_2011_2014.csv"
STATIONS_CACHE_FILE = CACHE_DIR / "dmi_precip_stations.csv"
SITE_STATION_CACHE_FILE = CACHE_DIR / "site_station_matches.csv"
DAILY_RAIN_CACHE_FILE = CACHE_DIR / "daily_rain_by_station_2011_2014.csv"

ANALYSIS_PANEL_FILE = OUTPUT_DIR / "analysis_panel_daily.csv"

DMI_CLIMATE_STATIONS_URL = "https://opendataapi.dmi.dk/v2/climateData/collections/station/items"
DMI_CLIMATE_VALUES_URL = "https://opendataapi.dmi.dk/v2/climateData/collections/stationValue/items"

ANALYSIS_START = "2011-01-01"
ANALYSIS_END = "2014-12-31"

DK_BBOX = "7,54,16,58"
RAIN_PARAMETER_ID = "acc_precip"

REQUEST_SLEEP_SEC = 0.15
MAX_RETRIES = 4
TIMEOUT = 60


def safe_request_json(url: str, params: dict | None = None) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            time.sleep(REQUEST_SLEEP_SEC)
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep((2 ** attempt) * 0.75)
    raise RuntimeError(f"Failed request: {url} params={params}") from last_err


def fetch_all_features(url: str, params: dict | None = None, limit: int = 1000) -> list[dict]:
    params = dict(params or {})
    params["limit"] = limit
    offset = 0
    features: list[dict] = []

    while True:
        page_params = dict(params)
        page_params["offset"] = offset
        payload = safe_request_json(url, page_params)
        page_features = payload.get("features", [])
        features.extend(page_features)

        if len(page_features) == 0 or len(page_features) < limit:
            break
        offset += limit

    return features


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_danish_price_csv(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", quotechar='"', encoding="utf-8-sig")
    df.columns = [c.strip().replace('"', "") for c in df.columns]
    df["Dato"] = pd.to_datetime(df["Dato"], dayfirst=True, errors="coerce")
    df["Pris"] = pd.to_numeric(df["Pris"], errors="coerce")
    df = df.rename(columns={"Dato": "date", "Pris": value_name})
    return df[["date", value_name]].dropna().sort_values("date").reset_index(drop=True)


def parse_cpi_file(path: Path) -> pd.DataFrame:
    # Statistics Denmark export with header text in row 1 and time columns in row 3.
    raw = pd.read_csv(path, header=None, skiprows=2, encoding="utf-8-sig")
    raw = raw.dropna(how="all").reset_index(drop=True)

    header = [str(v).strip() if pd.notna(v) else "" for v in raw.iloc[0].tolist()]
    data = raw.iloc[2:].reset_index(drop=True).copy()

    # Build column names by position so duplicate labels in the source file do not break selection.
    col_names: list[str] = []
    seen: dict[str, int] = {}
    for idx in range(data.shape[1]):
        if idx == 0:
            base = "unit"
        elif idx == 1:
            base = "series_name"
        else:
            candidate = header[idx] if idx < len(header) else ""
            base = candidate if candidate else f"col_{idx}"

        count = seen.get(base, 0)
        col_names.append(base if count == 0 else f"{base}_{count}")
        seen[base] = count + 1

    data.columns = col_names
    data["series_name"] = data["series_name"].astype(str).str.strip()

    value_cols = [c for c in data.columns if isinstance(c, str) and c.endswith(tuple(f"M{m:02d}" for m in range(1, 13)))]
    data = data[["series_name"] + value_cols].copy()

    mapping = {
        "00 Consumer price index, total": "cpi_total",
        "04.5.3 Liquid fuels": "cpi_liquid_fuels",
    }
    data = data[data["series_name"].isin(mapping.keys())].copy()
    if data.empty:
        raise ValueError("Could not find CPI series in CPI file.")

    data["series_key"] = data["series_name"].map(mapping)

    long = data.melt(
        id_vars=["series_name", "series_key"],
        value_vars=value_cols,
        var_name="year_month",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    ym = long["year_month"].astype(str).str.extract(r"^(?P<year>\d{4})M(?P<month>\d{2})$")
    long["date"] = pd.to_datetime(
        ym["year"] + "-" + ym["month"] + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )
    long = long[long["date"].notna()].copy()

    wide = long.pivot_table(index="date", columns="series_key", values="value", aggfunc="first").reset_index()

    wide["year"] = wide["date"].dt.year
    wide["month"] = wide["date"].dt.month
    return wide.sort_values("date").reset_index(drop=True)


def load_one_traffic_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=10)

    rename_map = {}
    for c in df.columns:
        if c == "(UTM32)":
            rename_map[c] = "x_utm32"
        elif c == "(UTM32).1":
            rename_map[c] = "y_utm32"
        elif c == "Dato":
            rename_map[c] = "date"
        elif c == "Vej-Id":
            rename_map[c] = "road_id_raw"
        elif c == "Vejnavn":
            rename_map[c] = "road_name"
        elif c == "Spor":
            rename_map[c] = "lane"

    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["x_utm32"] = pd.to_numeric(df["x_utm32"], errors="coerce")
    df["y_utm32"] = pd.to_numeric(df["y_utm32"], errors="coerce")
    df = df[df["date"].notna() & df["x_utm32"].notna() & df["y_utm32"].notna()].copy()

    hour_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("kl.")]
    if not hour_cols:
        raise ValueError(f"No hourly traffic columns found in file: {path}")

    for c in hour_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["daily_traffic"] = df[hour_cols].sum(axis=1, min_count=1)
    df["road_site_id"] = (
        df["road_id_raw"].astype(str).str.strip() + "||" +
        df["road_name"].astype(str).str.strip() + "||" +
        df["x_utm32"].round(0).astype(int).astype(str) + "||" +
        df["y_utm32"].round(0).astype(int).astype(str)
    )

    keep = ["road_site_id", "road_id_raw", "road_name", "lane", "x_utm32", "y_utm32", "date", "daily_traffic"]
    return df[keep].copy()


def load_all_traffic(paths: Iterable[Path] = TRAFFIC_FILES) -> pd.DataFrame:
    parts = [load_one_traffic_file(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    return df[(df["date"] >= ANALYSIS_START) & (df["date"] <= ANALYSIS_END)].copy()


def build_danish_holiday_table(start_year: int, end_year: int) -> pd.DataFrame:
    dk_holidays = holidays.DK(years=range(start_year, end_year + 1))
    hol_df = pd.DataFrame(
        {
            "holiday_date": pd.to_datetime(list(dk_holidays.keys())),
            "holiday_name": list(dk_holidays.values()),
        }
    ).sort_values("holiday_date")
    hol_df["day_before_holiday"] = hol_df["holiday_date"] - pd.Timedelta(days=1)
    hol_df["holiday_weekday"] = hol_df["holiday_date"].dt.day_name()
    hol_df["preholiday_weekday"] = hol_df["day_before_holiday"].dt.day_name()
    return hol_df


def get_or_create_holiday_table(start_year: int, end_year: int) -> pd.DataFrame:
    if HOLIDAY_CACHE_FILE.exists():
        return pd.read_csv(HOLIDAY_CACHE_FILE, parse_dates=["holiday_date", "day_before_holiday"])
    hol_df = build_danish_holiday_table(start_year, end_year)
    hol_df.to_csv(HOLIDAY_CACHE_FILE, index=False)
    return hol_df


def add_calendar_flags(df: pd.DataFrame, hol_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if hol_df is None:
        hol_df = get_or_create_holiday_table(int(df["date"].dt.year.min()), int(df["date"].dt.year.max()))

    holiday_set = set(hol_df["holiday_date"].dt.normalize())
    preholiday_set = set(hol_df["day_before_holiday"].dt.normalize())

    df = df.copy()
    df["date_norm"] = df["date"].dt.normalize()
    df["weekday"] = df["date"].dt.dayofweek
    df["weekday_name"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    df["day_type"] = np.select(
        [
            df["date_norm"].isin(holiday_set),
            df["date_norm"].isin(preholiday_set),
            df["date"].dt.dayofweek == 5,
            df["date"].dt.dayofweek == 6,
        ],
        [
            "holiday",
            "preholiday",
            "saturday",
            "sunday",
        ],
        default="weekday",
    )

    df["is_holiday"] = df["day_type"].eq("holiday")
    df["is_preholiday"] = df["day_type"].eq("preholiday")
    return df


def drop_holidays_and_preholidays(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~(df["is_holiday"] | df["is_preholiday"])].copy()


def add_wgs84_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["x_utm32"].values, df["y_utm32"].values)
    out = df.copy()
    out["lon"] = lon
    out["lat"] = lat
    return out


def fetch_dmi_precip_stations() -> pd.DataFrame:
    station_frames = []

    for station_type in ["Pluvio", "Synop"]:
        feats = fetch_all_features(
            DMI_CLIMATE_STATIONS_URL,
            params={"bbox": DK_BBOX, "type": station_type},
            limit=1000,
        )

        rows = []
        for f in feats:
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            coords = geom.get("coordinates", [None, None])

            rows.append(
                {
                    "station_id": str(props.get("stationId")),
                    "station_name": props.get("name"),
                    "station_type": props.get("type", station_type),
                    "station_status": props.get("status"),
                    "station_lon": coords[0],
                    "station_lat": coords[1],
                }
            )

        station_frames.append(pd.DataFrame(rows))

    stations = pd.concat(station_frames, ignore_index=True).drop_duplicates(subset=["station_id"])
    stations = stations[stations["station_id"].notna() & stations["station_lon"].notna() & stations["station_lat"].notna()].copy()
    return stations.reset_index(drop=True)


def get_or_create_stations() -> pd.DataFrame:
    if STATIONS_CACHE_FILE.exists():
        out = pd.read_csv(STATIONS_CACHE_FILE)
        out["station_id"] = out["station_id"].astype(str)
        return out
    stations = fetch_dmi_precip_stations()
    stations.to_csv(STATIONS_CACHE_FILE, index=False)
    return stations


def fetch_station_daily_precip(station_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    dt_range = f"{start_dt.strftime('%Y-%m-%dT00:00:00Z')}/{end_dt.strftime('%Y-%m-%dT23:59:59Z')}"
    feats = fetch_all_features(
        DMI_CLIMATE_VALUES_URL,
        params={
            "stationId": station_id,
            "parameterId": RAIN_PARAMETER_ID,
            "timeResolution": "day",
            "datetime": dt_range,
            "sortorder": "from,DESC",
        },
        limit=10000,
    )

    rows = []
    for f in feats:
        props = f.get("properties", {})
        rows.append(
            {
                "station_id": str(props.get("stationId")),
                "from": props.get("from"),
                "rain_mm_day": props.get("value"),
                "qc_status": props.get("qcStatus"),
                "validity": props.get("validity"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["from"] = pd.to_datetime(out["from"], errors="coerce", utc=True)
    out["date"] = out["from"].dt.tz_convert(None).dt.normalize()
    out["rain_mm_day"] = pd.to_numeric(out["rain_mm_day"], errors="coerce")
    out["rained_day"] = (out["rain_mm_day"].fillna(0) > 0).astype(int)
    return out[["station_id", "date", "rain_mm_day", "rained_day", "qc_status", "validity"]]


def station_has_precip_data(station_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> bool:
    dt_range = f"{start_dt.strftime('%Y-%m-%dT00:00:00Z')}/{end_dt.strftime('%Y-%m-%dT23:59:59Z')}"
    payload = safe_request_json(
        DMI_CLIMATE_VALUES_URL,
        params={
            "stationId": station_id,
            "parameterId": RAIN_PARAMETER_ID,
            "timeResolution": "day",
            "datetime": dt_range,
            "sortorder": "from,DESC",
            "limit": 1,
            "offset": 0,
        },
    )
    return len(payload.get("features", [])) > 0


def choose_best_station_for_site(
    site_lon: float,
    site_lat: float,
    stations: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    station_data_cache: dict[str, bool],
) -> tuple[str | None, float | None]:
    st = stations.copy()
    st["dist_km"] = st.apply(
        lambda r: haversine_km(site_lon, site_lat, r["station_lon"], r["station_lat"]),
        axis=1,
    )
    st = st.sort_values("dist_km").head(5)

    for _, row in st.iterrows():
        sid = str(row["station_id"])
        try:
            if sid not in station_data_cache:
                station_data_cache[sid] = station_has_precip_data(sid, start_dt, end_dt)
            if station_data_cache[sid]:
                return sid, float(row["dist_km"])
        except Exception as err:
            print(f"Warning: failed availability check for station {sid}: {err}")
            station_data_cache[sid] = False
    return None, None


def map_sites_to_stations(traffic_sites: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    unique_sites = traffic_sites[["road_site_id", "lon", "lat"]].drop_duplicates().copy()
    start_dt = pd.to_datetime(ANALYSIS_START)
    end_dt = pd.to_datetime(ANALYSIS_END)
    station_data_cache: dict[str, bool] = {}

    rows = []
    for _, row in unique_sites.iterrows():
        sid, dist = choose_best_station_for_site(
            site_lon=row["lon"],
            site_lat=row["lat"],
            stations=stations,
            start_dt=start_dt,
            end_dt=end_dt,
            station_data_cache=station_data_cache,
        )
        rows.append({"road_site_id": row["road_site_id"], "station_id": sid, "station_distance_km": dist})
    return pd.DataFrame(rows)


def extract_valid_station_ids(values: Iterable[object]) -> list[str]:
    valid: set[str] = set()
    for val in values:
        if pd.isna(val):
            continue
        sid = str(val).strip()
        if sid and sid.lower() not in {"nan", "none"}:
            valid.add(sid)
    return sorted(valid)


def get_or_create_site_station_matches(traffic: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    current_sites = traffic[["road_site_id", "lon", "lat"]].drop_duplicates().copy()

    if SITE_STATION_CACHE_FILE.exists():
        cached = pd.read_csv(SITE_STATION_CACHE_FILE)

        unresolved_mask = (
            cached["station_id"].isna()
            | cached["station_id"].astype(str).str.strip().eq("")
            | cached["station_id"].astype(str).str.lower().isin({"nan", "none"})
        )
        unresolved_ids = set(cached.loc[unresolved_mask, "road_site_id"])

        missing_ids = set(current_sites["road_site_id"]) - set(cached["road_site_id"])
        to_rematch_ids = missing_ids | unresolved_ids

        if not to_rematch_ids:
            return cached

        rematch_sites = current_sites[current_sites["road_site_id"].isin(to_rematch_ids)].copy()
        added = map_sites_to_stations(rematch_sites, stations)
        merged = pd.concat([cached, added], ignore_index=True).drop_duplicates(subset=["road_site_id"], keep="last")
        merged.to_csv(SITE_STATION_CACHE_FILE, index=False)
        return merged

    matches = map_sites_to_stations(current_sites, stations)
    matches.to_csv(SITE_STATION_CACHE_FILE, index=False)
    return matches


def fetch_daily_weather_for_needed_stations(station_ids: list[str], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for sid in station_ids:
        df = fetch_station_daily_precip(str(sid), start_dt, end_dt)
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def get_or_create_daily_rain(station_ids: list[str], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    station_ids = extract_valid_station_ids(station_ids)
    start_dt = pd.to_datetime(start_dt).normalize()
    end_dt = pd.to_datetime(end_dt).normalize()

    if DAILY_RAIN_CACHE_FILE.exists():
        try:
            rain = pd.read_csv(DAILY_RAIN_CACHE_FILE, parse_dates=["date"])
        except EmptyDataError:
            rain = pd.DataFrame()

        required_cols = {"station_id", "date", "rain_mm_day", "rained_day"}
        if rain.empty or not required_cols.issubset(set(rain.columns)):
            rain = fetch_daily_weather_for_needed_stations(station_ids, start_dt, end_dt)
            if not rain.empty:
                rain["station_id"] = rain["station_id"].astype(str)
                rain["date"] = pd.to_datetime(rain["date"]).dt.normalize()
            rain.to_csv(DAILY_RAIN_CACHE_FILE, index=False)
            return rain

        rain["station_id"] = rain["station_id"].astype(str)
        rain["date"] = rain["date"].dt.normalize()

        needed = rain[rain["station_id"].isin(station_ids) & rain["date"].between(start_dt, end_dt)].copy()

        coverage = needed.groupby("station_id")["date"].agg(["min", "max"])
        missing_station_ids = []
        for sid in station_ids:
            if sid not in coverage.index:
                missing_station_ids.append(sid)
                continue
            sid_min = coverage.loc[sid, "min"]
            sid_max = coverage.loc[sid, "max"]
            if sid_min > start_dt or sid_max < end_dt:
                missing_station_ids.append(sid)

        if missing_station_ids:
            fetched = fetch_daily_weather_for_needed_stations(missing_station_ids, start_dt, end_dt)
            if not fetched.empty:
                fetched["station_id"] = fetched["station_id"].astype(str)
                fetched["date"] = pd.to_datetime(fetched["date"]).dt.normalize()
                rain = pd.concat([rain, fetched], ignore_index=True)
                rain = rain.drop_duplicates(subset=["station_id", "date"], keep="last")
                rain.to_csv(DAILY_RAIN_CACHE_FILE, index=False)
                needed = rain[rain["station_id"].isin(station_ids) & rain["date"].between(start_dt, end_dt)].copy()
        return needed

    rain = fetch_daily_weather_for_needed_stations(station_ids, start_dt, end_dt)
    if not rain.empty:
        rain["station_id"] = rain["station_id"].astype(str)
        rain["date"] = pd.to_datetime(rain["date"]).dt.normalize()
    rain.to_csv(DAILY_RAIN_CACHE_FILE, index=False)
    return rain


def load_fuel_data() -> pd.DataFrame:
    gas = parse_danish_price_csv(FUEL_GAS_FILE, "gasoline_price")
    diesel = parse_danish_price_csv(FUEL_DIESEL_FILE, "diesel_price")
    fuel = gas.merge(diesel, on="date", how="outer").sort_values("date").reset_index(drop=True)
    fuel["year"] = fuel["date"].dt.year
    fuel["month"] = fuel["date"].dt.month
    return fuel


def add_real_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cpi_total" not in out.columns:
        raise ValueError("cpi_total is missing; cannot create inflation-adjusted fuel prices.")

    out["real_gasoline_price"] = np.where(
        out["gasoline_price"].notna() & out["cpi_total"].notna() & (out["cpi_total"] > 0),
        out["gasoline_price"] / out["cpi_total"] * 100,
        np.nan,
    )
    out["real_diesel_price"] = np.where(
        out["diesel_price"].notna() & out["cpi_total"].notna() & (out["cpi_total"] > 0),
        out["diesel_price"] / out["cpi_total"] * 100,
        np.nan,
    )

    out["log_traffic"] = np.where(out["daily_traffic"] > 0, np.log(out["daily_traffic"]), np.nan)
    out["log_gasoline_price"] = np.where(out["gasoline_price"] > 0, np.log(out["gasoline_price"]), np.nan)
    out["log_diesel_price"] = np.where(out["diesel_price"] > 0, np.log(out["diesel_price"]), np.nan)
    out["log_real_gasoline_price"] = np.where(out["real_gasoline_price"] > 0, np.log(out["real_gasoline_price"]), np.nan)
    out["log_real_diesel_price"] = np.where(out["real_diesel_price"] > 0, np.log(out["real_diesel_price"]), np.nan)

    return out
