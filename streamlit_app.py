# -*- coding: utf-8 -*-
# ==========================================
# 주제: 온실가스·이산화탄소·전지구 평균기온 — 공개 API 기반 대시보드
# 구성:
#  - 서론
#  - 꺾은선: 전세계 CO₂ 배출량(연), 대기 중 CO₂ 농도(월), 전지구 평균기온 이상(월)
#  - 지도: 국가별 온도 상승(연도 슬라이더)
# 데이터 소스(모두 무료/공개):
#  - OWID CO₂ 데이터(연): https://github.com/owid/co2-data (raw CSV)
#  - NOAA GML Mauna Loa CO₂(월): https://gml.noaa.gov/ccgg/trends/
#  - NASA GISTEMP v4(월): https://data.giss.nasa.gov/gistemp/
#  - Berkeley Earth 국가별 온도 변화(연): 정리본(OWID 데이터셋, raw CSV)
# ==========================================

from __future__ import annotations

import io
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------- 페이지 설정 ----------
st.set_page_config(page_title="온실가스·CO₂·기온: 공개데이터 대시보드", layout="wide")

# ---------- 상수 ----------
KST = ZoneInfo("Asia/Seoul")
TODAY_LOCAL = datetime.now(KST).date()

# ---------- 유틸 ----------
@st.cache_data(ttl=60 * 60)
def robust_get(url: str, params: dict | None = None, retry: int = 3, timeout: int = 20):
    last_err: Optional[Exception] = None
    for i in range(retry):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(1.25 * (i + 1))
    raise last_err if last_err else RuntimeError("Unknown request error")


def drop_future(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.date
    return out[out[date_col] <= TODAY_LOCAL]


# ---------- 데이터 로더 ----------
@st.cache_data(ttl=60 * 60)
def load_owid_co2_emissions_world() -> pd.DataFrame:
    """OWID CO₂ 데이터에서 세계(OWID_WRL) 연간 배출량(MtCO2) 로드"""
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    resp = robust_get(url)
    df = pd.read_csv(io.StringIO(resp.text))
    # 세계 합계: iso_code == OWID_WRL
    w = df[df["iso_code"] == "OWID_WRL"][["year", "co2"]].dropna()
    w = w.rename(columns={"year": "Year", "co2": "CO2 (Mt)"})
    return w.reset_index(drop=True)


@st.cache_data(ttl=60 * 60)
def load_noaa_mlo_co2_monthly() -> pd.DataFrame:
    """NOAA Mauna Loa 월별 CO₂ 농도(ppm) 로드"""
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    resp = robust_get(url)
    # 주석(#) 라인 스킵
    df = pd.read_csv(io.StringIO(resp.text), comment="#", header=None)
    # 문서 기준 컬럼: year, month, decimal_date, average, interpolated, trend, days
    df = df.rename(
        columns={
            0: "year",
            1: "month",
            2: "decimal_date",
            3: "average",
            4: "interpolated",
            5: "trend",
            6: "days",
        }
    )
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str)
        + "-"
        + df["month"].astype(int).astype(str).str.zfill(2)
        + "-15"
    )
    df["co2_ppm"] = pd.to_numeric(df["trend"], errors="coerce")  # 결측 시 trend 사용 권장
    out = (
        df.dropna(subset=["co2_ppm"])[["date", "co2_ppm"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    return drop_future(out, "date")


@st.cache_data(ttl=60 * 60)
def load_nasa_gistemp_monthly() -> pd.DataFrame:
    """NASA GISTEMP v4 전지구 월별 기온 이상(°C) 로드 → long(date,value)"""
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    resp = robust_get(url)
    raw = resp.text
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
    long = df.melt(id_vars=["Year"], var_name="month", value_name="value")
    month_map = {
        m[:3].capitalize(): i
        for i, m in enumerate(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            start=1,
        )
    }
    long["month"] = long["month"].astype(str).str[:3].str.capitalize().map(month_map)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["month", "value"])  # type: ignore
    long["date"] = pd.to_datetime(
        long["Year"].astype(int).astype(str)
        + "-"
        + long["month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    out = long[["date", "value"]].sort_values("date").reset_index(drop=True)
    return drop_future(out, "date")


@st.cache_data(ttl=60 * 60)
def load_country_temperature_change() -> pd.DataFrame:
    """Berkeley Earth 기반 국가별 연도별 온도 변화(°C). OWID 정리본 사용.
    컬럼 예시: Entity, Code(ISO3), Year, Temperature change from 1850-1900 (°C)
    """
    url = (
        "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Temperature%20change%20-%20Berkeley%20Earth/"
        "Temperature%20change%20-%20Berkeley%20Earth.csv"
    )
    resp = robust_get(url)
    df = pd.read_csv(io.StringIO(resp.text))
    # 유연한 컬럼 탐색
    col_entity = next((c for c in df.columns if c.lower() in ("entity", "country", "location")), None)
    col_code = next((c for c in df.columns if c.lower() in ("code", "iso3", "iso_code")), None)
    col_year = next((c for c in df.columns if c.lower() == "year"), None)
    value_cols = [c for c in df.columns if c not in (col_entity, col_code, col_year)]
    numeric_candidates = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not (col_entity and col_code and col_year and numeric_candidates):
        raise RuntimeError("Unexpected columns in temperature change dataset")
    value_col = numeric_candidates[-1]
    slim = df[[col_entity, col_code, col_year, value_col]].rename(
        columns={col_entity: "entity", col_code: "code", col_year: "year", value_col: "temp_change"}
    )
    slim = slim.dropna(subset=["code", "year", "temp_change"]).reset_index(drop=True)
    slim["year"] = slim["year"].astype(int)
    return slim


# ---------- 레이아웃 ----------
st.title("🌍 온실가스·CO₂·전지구 평균기온 — 공개데이터 대시보드")

st.markdown("## 서론")
st.write(
    "온실가스 배출량은 꾸준히 증가해 왔고, 대기 중 이산화탄소 농도 역시 상승 추세를 보입니다. "
    "이에 따라 전지구 평균기온(기온 이상)도 장기적으로 상승하고 있습니다. 아래의 공개 데이터 기반 꺾은선 그래프는 이러한 추세를 한눈에 보여주며, "
    "지도는 국가별 온도 변화 정도를 연도별로 탐색할 수 있도록 제공합니다."
)

# ----- 꺾은선: CO2 배출량(연) / 대기 CO2(월) / 전지구 평균기온 이상(월)
col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("전세계 CO₂ 배출량 (연)")
    try:
        df_emis = load_owid_co2_emissions_world()
        fig_emis = px.line(df_emis, x="Year", y="CO2 (Mt)", title="전세계 CO₂ 배출량 (MtCO₂)")
        st.plotly_chart(fig_emis, use_container_width=True)
    except Exception as e:
        st.warning(f"CO₂ 배출량 로드 실패: {e}")

    st.subheader("전지구 평균기온 이상 (월)")
    try:
        df_temp = load_nasa_gistemp_monthly()
        fig_temp = px.line(df_temp.tail(600), x="date", y="value", title="전지구 평균기온 이상(°C, NASA GISTEMP)")
        st.plotly_chart(fig_temp, use_container_width=True)
    except Exception as e:
        st.warning(f"기온 이상 로드 실패: {e}")

with col2:
    st.subheader("대기 중 CO₂ 농도 (월, 마우나 로아)")
    try:
        df_co2 = load_noaa_mlo_co2_monthly()
        fig_co2 = px.line(df_co2.tail(1200), x="date", y="co2_ppm", title="대기 중 CO₂ 농도(ppm, NOAA GML)")
        st.plotly_chart(fig_co2, use_container_width=True)
    except Exception as e:
        st.warning(f"대기 CO₂ 로드 실패: {e}")

st.markdown("---")

# ----- 지도: 국가별 온도 변화(연도 슬라이더)
st.subheader("국가별 온도 변화 (연도별 탐색)")
try:
    df_ct = load_country_temperature_change()
    years = sorted(df_ct["year"].unique())
    default_year = years[-1]
    pick = st.slider("연도 선택", min_value=int(years[0]), max_value=int(years[-1]), value=int(default_year), step=1)

    focus = df_ct[df_ct["year"] == int(pick)].copy()
    # 색상 범위: 극단값 완화
    vmin = float(np.nanpercentile(focus["temp_change"], 5))
    vmax = float(np.nanpercentile(focus["temp_change"], 95))
    fig_map = px.choropleth(
        focus,
        locations="code",
        color="temp_change",
        hover_name="entity",
        color_continuous_scale="RdBu_r",
        range_color=(vmin, vmax),
        title=f"전세계 온도 변화(°C) — {pick}년",
    )
    fig_map.update_layout(coloraxis_colorbar=dict(title="온도 변화 (°C)"))
    st.plotly_chart(fig_map, use_container_width=True)
except Exception as e:
    st.warning(f"국가별 온도 변화 지도 로드 실패: {e}")

st.markdown(
    """
---
데이터 출처
- CO₂ 배출량(연): Our World in Data CO₂ dataset (OWID_WRL), https://github.com/owid/co2-data
- 대기 CO₂(월): NOAA GML Mauna Loa, https://gml.noaa.gov/ccgg/trends/
- 전지구 평균기온 이상(월): NASA GISTEMP v4, https://data.giss.nasa.gov/gistemp/
- 국가별 온도 변화(연): Berkeley Earth (OWID 정리본)
"""
)
