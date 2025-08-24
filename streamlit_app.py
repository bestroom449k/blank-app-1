# streamlit_app.py
# ------------------------------------------------------------
# "날씨 알리미" (Korean UI) — Streamlit 실시간/예보 날씨 앱
# - 데이터 소스: Open-Meteo(무료/오픈소스, API 키 불필요)
# - UI 오픈소스: Leaflet(지도, folium/streamlit-folium), Plotly(차트)
# - Python 3.10+, GitHub Codespaces/로컬 모두 동작
# - 외부 의존성 최소(핵심만 사용), 모든 핵심 로직 주석 포함
# ------------------------------------------------------------

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import pytz
import streamlit as st
from datetime import datetime, timedelta, timezone

# 오픈소스 UI 라이브러리(지도/차트)
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ------------------------------------------------------------
# 기본 페이지 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="날씨 알리미 · Streamlit",
    page_icon="🌤️",
    layout="wide",
)

# ------------------------------------------------------------
# 유틸: Open-Meteo 날씨코드 → 한국어 설명/이모지
# (공식 코드표 요약본)
# ------------------------------------------------------------
WEATHERCODE_KR = {
    0: ("맑음", "☀️"),
    1: ("대체로 맑음", "🌤️"),
    2: ("부분적으로 흐림", "⛅"),
    3: ("흐림", "☁️"),
    45: ("안개", "🌫️"),
    48: ("착빙 안개", "🌫️"),
    51: ("약한 이슬비", "🌦️"),
    53: ("보통 이슬비", "🌦️"),
    55: ("강한 이슬비", "🌧️"),
    56: ("약한 냉이슬비", "🌧️"),
    57: ("강한 냉이슬비", "🌧️"),
    61: ("약한 비", "🌧️"),
    63: ("보통 비", "🌧️"),
    65: ("강한 비", "🌧️"),
    66: ("약한 냉비", "🌧️"),
    67: ("강한 냉비", "🌧️"),
    71: ("약한 눈", "❄️"),
    73: ("보통 눈", "❄️"),
    75: ("강한 눈", "❄️"),
    77: ("싸락눈", "❄️"),
    80: ("약한 소나기", "🌦️"),
    81: ("보통 소나기", "🌦️"),
    82: ("강한 소나기", "⛈️"),
    85: ("약한 소낙눈", "🌨️"),
    86: ("강한 소낙눈", "🌨️"),
    95: ("천둥번개(약~중)", "⛈️"),
    96: ("천둥·우박(약)", "⛈️"),
    99: ("천둥·우박(강)", "⛈️"),
}

def code_to_text_emoji(wcode: Optional[int]) -> Tuple[str, str]:
    return WEATHERCODE_KR.get(int(wcode) if wcode is not None else -1, ("정보 없음", "❔"))

# ------------------------------------------------------------
# 사이드바: 위치/옵션
# ------------------------------------------------------------
with st.sidebar:
    st.header("📍 위치 선택")
    # 빠른 선택(한국 주요 도시)
    quick = st.selectbox(
        "빠른 선택",
        [
            "서울 (37.5665, 126.9780)",
            "부산 (35.1796, 129.0756)",
            "대구 (35.8714, 128.6014)",
            "인천 (37.4563, 126.7052)",
            "광주 (35.1595, 126.8526)",
            "대전 (36.3504, 127.3845)",
            "울산 (35.5384, 129.3114)",
            "제주 (33.4996, 126.5312)",
        ],
        index=0,
        help="원하는 도시를 빠르게 고르세요. 아래에서 직접 검색도 가능합니다.",
    )

    st.markdown("**또는** 직접 검색")
    q = st.text_input(
        "도시/지역 이름(예: Tokyo, New York, 파리, 서울 마포구)",
        value="",
        help="Open-Meteo Geocoding(무료)을 사용해 최대 5개 후보를 찾습니다.",
    )
    search_btn = st.button("🔎 위치 검색")

    st.divider()
    st.header("⚙️ 옵션")
    unit_temp = st.radio("온도 단위", ["°C", "°F"], horizontal=True, index=0)
    forecast_days = st.slider("예보 일수", min_value=3, max_value=10, value=7)
    hours_to_show = st.slider("시간별 차트 표시 시간", min_value=12, max_value=72, value=36, step=6)
    tz_label = "Asia/Seoul"  # 기본 표시는 KST

# ------------------------------------------------------------
# 지오코딩: Open-Meteo Geocoding API
# - API 키 불필요, name과 언어/수 제한만 지정
# - 캐시로 과도한 호출 방지
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_search(query: str) -> List[Dict]:
    if not query.strip():
        return []
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 5, "language": "ko", "format": "json"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []

# 빠른 선택에서 lat/lon 추출
def parse_quick(s: str) -> Tuple[float, float, str]:
    # 예: "서울 (37.5665, 126.9780)"
    name = s.split(" (")[0]
    inside = s.split("(")[1].split(")")[0]
    lat_str, lon_str = [x.strip() for x in inside.split(",")]
    return float(lat_str), float(lon_str), name

# ------------------------------------------------------------
# 날씨 API 호출: Open-Meteo Forecast API
# - current/hourly/daily 한 번에 요청
# ------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_forecast(
    lat: float,
    lon: float,
    tz: str,
    temp_unit: str = "celsius",
    days: int = 7,
) -> Dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        # 현재 날씨
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "weathercode",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        # 시간별
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "weathercode",
            "wind_speed_10m",
        ]),
        # 일별
        "daily": ",".join([
            "weathercode",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
        ]),
        "forecast_days": days,
        "timezone": tz,
        "temperature_unit": "celsius" if temp_unit == "celsius" else "fahrenheit",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------
# 메인 로직: 선택 위치 확정
# ------------------------------------------------------------
selected_lat, selected_lon, selected_name = parse_quick(quick)

# 사용자가 검색 버튼을 누르면 지오코딩 결과 표시
if search_btn and q.strip():
    try:
        candidates = geocode_search(q)
        if candidates:
            st.sidebar.success(f"검색 결과: {len(candidates)}개")
            names = [
                f"{c.get('name')} ({c.get('country_code', '')}) · {c.get('latitude'):.4f}, {c.get('longitude'):.4f}"
                + (f" · {c.get('admin1')}" if c.get("admin1") else "")
                for c in candidates
            ]
            picked = st.sidebar.radio("검색 결과에서 선택", names, index=0)
            idx = names.index(picked)
            c = candidates[idx]
            selected_lat, selected_lon = float(c["latitude"]), float(c["longitude"])
            selected_name = f"{c.get('name')} ({c.get('country_code','')})"
        else:
            st.sidebar.warning("검색 결과가 없습니다. 표기(언어)나 철자를 바꿔보세요.")
    except Exception as e:
        st.sidebar.error(f"지오코딩 오류: {e}")

# ------------------------------------------------------------
# 데이터 가져오기
# ------------------------------------------------------------
try:
    temp_unit_key = "celsius" if unit_temp == "°C" else "fahrenheit"
    data = fetch_forecast(selected_lat, selected_lon, tz_label, temp_unit_key, forecast_days)
except Exception as e:
    st.error(f"날씨 API 요청 중 오류가 발생했습니다: {e}")
    st.stop()

# ------------------------------------------------------------
# 상단 헤더/메트릭
# ------------------------------------------------------------
st.title("🌤️ 날씨 알리미")
st.caption("Open-Meteo 데이터를 활용한 실시간/예보 날씨 — 지도(Leaflet)와 차트(Plotly) UI")

col_a, col_b, col_c = st.columns([2, 1, 1], vertical_alignment="center")
with col_a:
    st.subheader(f"📍 {selected_name}  ·  {selected_lat:.4f}, {selected_lon:.4f}")
with col_b:
    st.write("")
with col_c:
    st.write(f"표시 시간대: **{tz_label}**")

# 현재 날씨 블록
current = data.get("current", {})
w_txt, w_emoji = code_to_text_emoji(current.get("weathercode"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("현재 기온", f"{current.get('temperature_2m','?')} {unit_temp}")
c2.metric("체감 온도", f"{current.get('apparent_temperature','?')} {unit_temp}")
c3.metric("습도", f"{current.get('relative_humidity_2m','?')} %")
c4.metric("풍속", f"{current.get('wind_speed_10m','?')} km/h")
st.markdown(f"**현재 상태:** {w_emoji} {w_txt}")

# ------------------------------------------------------------
# 지도(Leaflet by folium)
# ------------------------------------------------------------
with st.expander("🗺️ 지도 보기 (Leaflet)", expanded=True):
    fmap = folium.Map(location=[selected_lat, selected_lon], zoom_start=10, control_scale=True)
    folium.Marker(
        [selected_lat, selected_lon],
        popup=f"{selected_name}",
        tooltip="선택 위치",
        icon=folium.Icon(color="blue", icon="cloud", prefix="fa"),
    ).add_to(fmap)
    st_folium(fmap, height=340, use_container_width=True)

# ------------------------------------------------------------
# 시간별/일별 데이터프레임으로 정리
# ------------------------------------------------------------
def to_df_hourly(d: Dict) -> pd.DataFrame:
    h = d.get("hourly", {})
    if not h:
        return pd.DataFrame()
    dfh = pd.DataFrame(h)
    dfh["time"] = pd.to_datetime(dfh["time"])
    return dfh

def to_df_daily(d: Dict) -> pd.DataFrame:
    daily = d.get("daily", {})
    if not daily:
        return pd.DataFrame()
    dfd = pd.DataFrame(daily)
    dfd["time"] = pd.to_datetime(dfd["time"])
    return dfd

df_hourly = to_df_hourly(data)
df_daily = to_df_daily(data)

# 가까운 시간 N시간만 표시(시간별)
if not df_hourly.empty:
    now = pd.Timestamp.now(tz=pytz.timezone(tz_label)).tz_localize(None)
    mask = (df_hourly["time"] >= now) & (df_hourly["time"] <= now + pd.Timedelta(hours=hours_to_show))
    view_hourly = df_hourly.loc[mask].reset_index(drop=True)
else:
    view_hourly = pd.DataFrame()

# ------------------------------------------------------------
# 시간별 차트(Plotly)
# ------------------------------------------------------------
st.subheader("⏱️ 시간별 예보")
if view_hourly.empty:
    st.info("표시할 시간별 데이터가 없습니다.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view_hourly["time"],
        y=view_hourly["temperature_2m"],
        mode="lines+markers",
        name=f"기온({unit_temp})",
    ))
    if "precipitation_probability" in view_hourly.columns:
        fig.add_trace(go.Bar(
            x=view_hourly["time"],
            y=view_hourly["precipitation_probability"],
            name="강수확률(%)",
            opacity=0.4,
            yaxis="y2",
        ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="시간",
        yaxis_title=f"기온({unit_temp})",
        yaxis2=dict(title="강수확률(%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("데이터 보기", expanded=False):
        st.dataframe(view_hourly, use_container_width=True, height=280)

# ------------------------------------------------------------
# 일별 요약(카드 + 차트)
# ------------------------------------------------------------
st.subheader("📅 일별 예보")
if df_daily.empty:
    st.info("표시할 일별 데이터가 없습니다.")
else:
    # 상단 카드들
    cc = st.columns(min(5, len(df_daily)))
    for i in range(min(len(df_daily), 5)):
        row = df_daily.iloc[i]
        txt, emo = code_to_text_emoji(row.get("weathercode"))
        with cc[i]:
            st.markdown(
                f"**{row['time'].strftime('%m/%d (%a)')}**  \n"
                f"{emo} {txt}  \n"
                f"최고 {row['temperature_2m_max']}{unit_temp} / 최저 {row['temperature_2m_min']}{unit_temp}  \n"
                f"예상 강수량 {row.get('precipitation_sum', 0)} mm"
            )

    # 범위 차트(최고/최저)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_daily["time"],
        y=df_daily["temperature_2m_max"],
        mode="lines+markers",
        name=f"최고기온({unit_temp})",
        line=dict(width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=df_daily["time"],
        y=df_daily["temperature_2m_min"],
        mode="lines+markers",
        name=f"최저기온({unit_temp})",
        line=dict(width=2, dash="dot"),
    ))
    if "precipitation_probability_max" in df_daily.columns:
        fig2.add_trace(go.Bar(
            x=df_daily["time"],
            y=df_daily["precipitation_probability_max"],
            name="최대 강수확률(%)",
            opacity=0.35,
            yaxis="y2",
        ))
    fig2.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="날짜",
        yaxis_title=f"기온({unit_temp})",
        yaxis2=dict(title="강수확률(%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("일별 데이터 보기", expanded=False):
        st.dataframe(df_daily, use_container_width=True, height=280)

# ------------------------------------------------------------
# 푸터/출처 표기
# ------------------------------------------------------------
st.caption("데이터: Open-Meteo (https://open-meteo.com) · 지도: Leaflet(through folium) · 차트: Plotly")
