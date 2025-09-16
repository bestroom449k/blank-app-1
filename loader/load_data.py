import os
import io
import time
from datetime import datetime, date

import pandas as pd  # type: ignore  # resolved in loader container (see loader/requirements.txt)
import requests
from dateutil import parser as dup
import psycopg2  # type: ignore  # resolved in loader container
from psycopg2.extras import execute_values  # type: ignore  # resolved in loader container

NASA_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

# Simple retry

def robust_get(url: str, params: dict | None = None, retry: int = 3, timeout: int = 20):
    last = None
    for i in range(retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last = e
        time.sleep(1.25 * (i + 1))
    if last:
        raise last
    raise RuntimeError("Unknown request error")


def load_nasa_monthly() -> pd.DataFrame:
    raw = robust_get(NASA_URL).text
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    header_idx = 0
    for i, ln in enumerate(lines):
        if ln.lower().startswith("year"):
            header_idx = i
            break
    clean = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(clean))
    for col in ["J-D", "D-N", "DJF", "MAM", "JJA", "SON"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df_long = df.melt(id_vars=["Year"], var_name="month", value_name="value")
    month_map = {m[:3].capitalize(): i for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
    df_long["month"] = df_long["month"].str[:3].str.capitalize().map(month_map)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.dropna(subset=["month", "value"])  # type: ignore
    df_long["date"] = pd.to_datetime(
        df_long["Year"].astype(int).astype(str)
        + "-"
        + df_long["month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    out = df_long[["date", "value"]].sort_values("date").reset_index(drop=True)
    out["metric"] = "global_temp_anomaly_c"
    return out


def load_worldbank_indicator(countries: list[str], indicator: str) -> pd.DataFrame:
    base = "https://api.worldbank.org/v2/country/{}/indicator/{}"
    url = base.format(";".join([c.lower() for c in countries]), indicator)
    params = dict(format="json", per_page=20000)
    js = robust_get(url, params=params).json()
    if not isinstance(js, list) or len(js) < 2 or js[1] is None:
        raise RuntimeError("Unexpected World Bank response")
    data = js[1]
    rows = []
    for row in data:
        val = row.get("value")
        yr = row.get("date")
        c = row.get("country", {}).get("id") or row.get("countryiso3code")
        if val is None or yr is None or c is None:
            continue
        rows.append(dict(country=c, value=float(val), date=pd.Timestamp(int(yr), 1, 1)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Empty World Bank data for indicator " + indicator)
    df["indicator"] = indicator
    return df


def get_conn():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME", "appdb"),
        user=os.environ.get("DB_USER", "appuser"),
        password=os.environ.get("DB_PASSWORD", "apppassword"),
    )
    conn.autocommit = True
    return conn


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists nasa_gistemp_monthly (
                date date primary key,
                value double precision not null,
                metric text not null
            );
            create table if not exists worldbank_indicators (
                country text not null,
                indicator text not null,
                date date not null,
                value double precision not null,
                primary key (country, indicator, date)
            );
            create or replace view vw_pisa_vs_temp as
            with annual_temp as (
                select extract(year from date)::int as year,
                       avg(value) as temp_anom
                from nasa_gistemp_monthly
                group by 1
            )
            select wb.country,
                   wb.indicator,
                   extract(year from wb.date)::int as year,
                   wb.value as pisa_score,
                   at.temp_anom
            from worldbank_indicators wb
            left join annual_temp at
              on at.year = extract(year from wb.date)::int;
            """
        )


def upsert_nasa(conn, df: pd.DataFrame):
    def _to_date(x):
        if isinstance(x, pd.Timestamp):
            return x.date()
        if isinstance(x, datetime):
            return x.date()
        if isinstance(x, date):
            return x
        return pd.to_datetime(x).date()
    records = df[["date", "value", "metric"]].to_dict("records")
    values = [(_to_date(r["date"]), float(r["value"]), str(r["metric"])) for r in records]
    with conn.cursor() as cur:
        execute_values(cur,
            """
            insert into nasa_gistemp_monthly (date, value, metric)
            values %s
            on conflict (date) do update set value = excluded.value, metric = excluded.metric
            """,
            values,
            page_size=1000,
        )


def upsert_worldbank(conn, df: pd.DataFrame):
    def _to_date(x):
        if isinstance(x, pd.Timestamp):
            return x.date()
        if isinstance(x, datetime):
            return x.date()
        if isinstance(x, date):
            return x
        return pd.to_datetime(x).date()
    records = df[["country", "indicator", "date", "value"]].to_dict("records")
    values = [
        (str(r["country"]), str(r["indicator"]), _to_date(r["date"]), float(r["value"]))
        for r in records
    ]
    with conn.cursor() as cur:
        execute_values(cur,
            """
            insert into worldbank_indicators (country, indicator, date, value)
            values %s
            on conflict (country, indicator, date) do update set value = excluded.value
            """,
            values,
            page_size=1000,
        )


def main():
    countries = [c.strip().upper() for c in os.environ.get("WB_COUNTRIES", "KOR,JPN,USA").split(",") if c.strip()]
    indicators = [i.strip().upper() for i in os.environ.get("WB_INDICATORS", "LO.PISA.MAT").split(",") if i.strip()]

    conn = get_conn()
    ensure_schema(conn)

    # NASA
    nasa = load_nasa_monthly()
    upsert_nasa(conn, nasa)
    print(f"NASA rows: {len(nasa)}")

    # World Bank
    total = 0
    for ind in indicators:
        wb = load_worldbank_indicator(countries, ind)
        upsert_worldbank(conn, wb)
        total += len(wb)
        print(f"WB {ind} rows: {len(wb)}")

    print(f"Done. Total WB rows: {total}")


if __name__ == "__main__":
    main()
