# streamlit_app.py
# ------------------------------------------------------------
# "날씨 알리미" — 미려 UI & 애니메이션 적용 버전
# - UI: Animate.css(CDN), Streamlit Option Menu, 커스텀 CSS(글래스모피즘 카드)
# - 기능: 현지 시간대/현재 시각, 해외 도시 검색(ko→en 2단계), 데이터 한글화
# - 데이터: Open-Meteo (API Key 불필요)
# - Python 3.10+, Codespaces/로컬 모두 지원
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime

import requests
import pandas as pd
import pytz
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu

# ------------------------------------------------------------
# 페이지 & 글로벌 스타일
# ------------------------------------------------------------
st.set_page_config(
    page_title="날씨 알리미 · Streamlit",
    page_icon="🌤️",
    layout="wide",
)

# 애니메이션/테마 CSS (Animate.css + 글래스 카드 + 헤더 그라디언트)
st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
<style>
:root {
  --glass-bg: rgba(255,255,255,0.55);
  --glass-border: 1px solid rgba(255,255,255,0.3);
  --glass-blur: blur(10px);
}
.hero {
  position: relative;
  padding: 22px 24px;
  border-radius: 18px;
  background: linear-gradient(120deg, #6fb1fc, #4364f7, #1e3c72);
  color: #fff;
  overflow: hidden;
}
.hero:before {
  content: "";
  position: absolute;
  top: -40%; left: -20%;
  width: 70%; height: 200%;
  background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.35), transparent 60%);
  filter: blur(40px);
  animation: floaty 9s ease-in-out infinite alternate;
}
@keyframes floaty { from { transform: translateX(0px);} to { transform: translateX(30px);} }

.glass {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  border: var(--glass-border);
  border-radius: 16px;
  padding: 16px;
}
.metric-card {
  transition: transform .2s ease, box-shadow .2s ease;
}
.metric-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.2);
  border: 1px solid rgba(255,255,255,0.35);
  font-size: 0.9rem;
}
.small { font-size: 0.92rem; opacity: 0.95; }
hr.soft { border: none; height: 1px; background: rgba(0,0,0,0.06); margin: 8px 0 16px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 날씨 코드 → 한글/이모지 매핑
# ------------------------------------------------------------
WEATHERCODE_KR = {
    0: ("맑음", "☀️"), 1: ("대체로 맑음", "🌤️"), 2: ("부분적으로 흐림", "⛅"), 3: ("흐림", "☁️"),
    45: ("안개", "🌫️"), 48: ("착빙 안개", "🌫️"),
    51: ("약한 이슬비", "🌦️"), 53: ("보통 이슬비", "🌦️"), 55: ("강한 이슬비", "🌧️"),
    56: ("약한 냉이슬비", "🌧️"), 57: ("강한 냉이슬비", "🌧️"),
    61: ("약한 비", "🌧️"), 63: ("보통 비", "🌧️"), 65: ("강한 비", "🌧️"),
    66: ("약한 냉비", "🌧️"), 67: ("강한 냉비", "🌧️"),
    71: ("약한 눈", "❄️"), 73: ("보통 눈", "❄️"), 75: ("강한 눈", "❄️"),
    77: ("싸락눈", "❄️"),
    80: ("약한 소나기", "🌦️"), 81: ("보통 소나기", "🌦️"), 82: ("강한 소나기", "⛈️"),
    85: ("약한 소낙눈", "🌨️"), 86: ("강한 소낙눈", "🌨️"),
    95: ("천둥번개(약~중)", "⛈️"), 96: ("천둥·우박(약)", "⛈️"), 99: ("천둥·우박(강)", "⛈️"),
}
def code_to_text_emoji(w: Optional[int]) -> Tuple[str, str]:
    return WEATHERCODE_KR.get(int(w) if w is not None else -1, ("정보 없음", "❔"))

# ------------------------------------------------------------
# 사이드바 — 위치/옵션
# ------------------------------------------------------------
with st.sidebar:
    st.header("📍 위치 선택")
    quick = st.selectbox(
        "빠른 선택",
        [
            "서울 (37.5665, 126.9780)",
            "부산 (35.1796, 129.0756)",
            "도쿄 (35.6762, 139.6503)",
            "뉴욕 (40.7128, -74.0060)",
            "파리 (48.8566, 2.3522)",
            "런던 (51.5074, -0.1278)",
            "시드니 (-33.8688, 151.2093)",
            "싱가포르 (1.3521, 103.8198)",
        ],
        index=0,
        help="원하는 도시를 빠르게 고르세요. 아래에서 직접 검색도 가능합니다.",
    )

    st.markdown("**또는** 직접 검색")
    q = st.text_input("도시/지역 이름(예: Tokyo, New York, 파리, 서울 마포구)", value="")
    search_btn = st.button("🔎 위치 검색")

    st.divider()
    st.header("⚙️ 옵션")
    unit_temp = st.radio("온도 단위", ["°C", "°F"], horizontal=True, index=0)
    forecast_days = st.slider("예보 일수", min_value=3, max_value=10, value=7)
    hours_to_show = st.slider("시간별 차트 표시 시간", min_value=12, max_value=72, value=36, step=6)
    anim_on = st.toggle("애니메이션 효과", value=True)

# 검색 상태 유지용 세션
if "geo_results" not in st.session_state: st.session_state.geo_results = []
if "geo_pick" not in st.session_state: st.session_state.geo_pick = 0

def parse_quick(s: str) -> Tuple[float, float, str]:
    # "서울 (37.5665, 126.9780)" → (lat, lon, "서울")
    name = s.split(" (")[0]
    inside = s.split("(")[1].split(")")[0]
    lat_str, lon_str = [x.strip() for x in inside.split(",")]
    return float(lat_str), float(lon_str), name

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_search(query: str) -> List[Dict]:
    """한국어 → 영어 순차 검색으로 해외 검색 실패를 줄임."""
    if not query.strip(): return []
    base = "https://geocoding-api.open-meteo.com/v1/search"
    def _req(lang: str):
        r = requests.get(base, params={"name": query, "count": 5, "language": lang, "format": "json"}, timeout=15)
        r.raise_for_status()
        return r.json().get("results", []) or []
    res = _req("ko")
    return res if res else _req("en")

if search_btn:
    try:
        st.session_state.geo_results = geocode_search(q)
        st.session_state.geo_pick = 0
        if not st.session_state.geo_results:
            st.sidebar.warning("검색 결과가 없습니다. 표기(언어)나 철자를 바꿔보세요.")
    except Exception as e:
        st.sidebar.error(f"지오코딩 오류: {e}")

# 결과 선택
selected_lat, selected_lon, selected_name = parse_quick(quick)
if st.session_state.geo_results:
    names = [
        f"{c.get('name')} ({c.get('country_code','')}) · {c.get('latitude'):.4f}, {c.get('longitude'):.4f}"
        + (f" · {c.get('admin1')}" if c.get("admin1") else "")
        for c in st.session_state.geo_results
    ]
    st.sidebar.selectbox(
        "검색 결과에서 선택",
        options=list(range(len(names))),
        index=st.session_state.geo_pick,
        format_func=lambda i: names[i],
        key="geo_pick",
    )
    pick = st.session_state.geo_results[st.session_state.geo_pick]
    selected_lat, selected_lon = float(pick["latitude"]), float(pick["longitude"])
    selected_name = f"{pick.get('name')} ({pick.get('country_code','')})"

# ------------------------------------------------------------
# 예보 API (timezone='auto')
# ------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_forecast(lat: float, lon: float, temp_unit: str = "celsius", days: int = 7) -> Dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": ",".join([
            "temperature_2m","relative_humidity_2m","apparent_temperature",
            "is_day","precipitation","weathercode","wind_speed_10m","wind_direction_10m",
        ]),
        "hourly": ",".join([
            "temperature_2m","relative_humidity_2m","precipitation_probability",
            "precipitation","weathercode","wind_speed_10m",
        ]),
        "daily": ",".join([
            "weathercode","temperature_2m_max","temperature_2m_min",
            "precipitation_sum","precipitation_probability_max","wind_speed_10m_max",
        ]),
        "forecast_days": days,
        "timezone": "auto",
        "temperature_unit": "celsius" if temp_unit == "celsius" else "fahrenheit",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# 데이터 로드
try:
    temp_key = "celsius" if unit_temp == "°C" else "fahrenheit"
    data = fetch_forecast(selected_lat, selected_lon, temp_key, forecast_days)
except Exception as e:
    st.error(f"날씨 API 요청 중 오류가 발생했습니다: {e}")
    st.stop()

# ------------------------------------------------------------
# 상단 히어로 섹션 (애니메이션 헤더)
# ------------------------------------------------------------
tz_name = data.get("timezone", "UTC")
utc_offset = int(data.get("utc_offset_seconds", 0) // 3600)
now_local = datetime.now(pytz.timezone(tz_name)) if tz_name else datetime.utcnow().replace(tzinfo=pytz.utc)

header_anim_cls = "animate__animated animate__fadeInDown" if anim_on else ""
st.markdown(
    f"""
<div class="hero {header_anim_cls}">
  <h1 style="margin:8px 0 4px;">🌤️ 날씨 알리미</h1>
  <div class="small">{selected_name} · {selected_lat:.4f}, {selected_lon:.4f}</div>
  <div class="small">현지 시간대: <b>{tz_name}</b> · UTC{utc_offset:+d} · 현재 시각: <b>{now_local.strftime("%Y-%m-%d %H:%M")}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 상단 네비게이션 (Option Menu)
# ------------------------------------------------------------
choice = option_menu(
    None,
    ["개요", "시간별", "일별", "지도"],
    icons=["cloud-sun", "clock", "calendar-event", "map"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# ------------------------------------------------------------
# 공용: DataFrame 빌드 & 한글화
# ------------------------------------------------------------
def to_df_hourly(d: Dict) -> pd.DataFrame:
    h = d.get("hourly", {})
    if not h: return pd.DataFrame()
    dfh = pd.DataFrame(h)
    dfh["time"] = pd.to_datetime(dfh["time"])  # 로컬 시간(naive)
    return dfh

def to_df_daily(d: Dict) -> pd.DataFrame:
    dd = d.get("daily", {})
    if not dd: return pd.DataFrame()
    dfd = pd.DataFrame(dd)
    dfd["time"] = pd.to_datetime(dfd["time"])
    return dfd

def make_hourly_display(df: pd.DataFrame, unit_label: str) -> pd.DataFrame:
    if df.empty: return df
    disp = df.copy()
    disp["날씨"] = disp["weathercode"].apply(lambda c: f"{code_to_text_emoji(c)[1]} {code_to_text_emoji(c)[0]}")
    rename_map = {
        "time": "시간",
        "temperature_2m": f"기온({unit_label})",
        "relative_humidity_2m": "상대습도(%)",
        "precipitation_probability": "강수확률(%)",
        "precipitation": "강수량(mm)",
        "wind_speed_10m": "풍속(km/h)",
        "weathercode": "날씨코드",
    }
    disp.rename(columns={k:v for k,v in rename_map.items() if k in disp.columns}, inplace=True)
    order = ["시간","날씨",f"기온({unit_label})","상대습도(%)","강수확률(%)","강수량(mm)","풍속(km/h)","날씨코드"]
    return disp[[c for c in order if c in disp.columns]]

def make_daily_display(df: pd.DataFrame, unit_label: str) -> pd.DataFrame:
    if df.empty: return df
    disp = df.copy()
    disp["날씨"] = disp["weathercode"].apply(lambda c: f"{code_to_text_emoji(c)[1]} {code_to_text_emoji(c)[0]}")
    rename_map = {
        "time":"날짜",
        "temperature_2m_max": f"최고기온({unit_label})",
        "temperature_2m_min": f"최저기온({unit_label})",
        "precipitation_sum": "강수량합(mm)",
        "precipitation_probability_max": "최대 강수확률(%)",
        "wind_speed_10m_max": "최대 풍속(km/h)",
        "weathercode": "날씨코드",
    }
    disp.rename(columns={k:v for k,v in rename_map.items() if k in disp.columns}, inplace=True)
    order = ["날짜","날씨",f"최고기온({unit_label})",f"최저기온({unit_label})","최대 강수확률(%)","강수량합(mm)","최대 풍속(km/h)","날씨코드"]
    return disp[[c for c in order if c in disp.columns]]

df_hourly = to_df_hourly(data)
df_daily = to_df_daily(data)

# 현재 시각 이후 N시간(시간별)
try:
    now_local_naive = datetime.now(pytz.timezone(tz_name)).replace(tzinfo=None)
except Exception:
    now_local_naive = datetime.utcnow()
if not df_hourly.empty:
    mask = (df_hourly["time"] >= now_local_naive) & (df_hourly["time"] <= now_local_naive + pd.Timedelta(hours=hours_to_show))
    view_hourly = df_hourly.loc[mask].reset_index(drop=True)
else:
    view_hourly = pd.DataFrame()

# ------------------------------------------------------------
# 개요
# ------------------------------------------------------------
if choice == "개요":
    section_anim = "animate__animated animate__fadeInUp" if anim_on else ""
    current = data.get("current", {})
    w_txt, w_emo = code_to_text_emoji(current.get("weathercode"))

    st.markdown(f'<div class="glass {section_anim}">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("#### 현재 상태")
        st.markdown(f"### {w_emo} {w_txt}")
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("현재 기온", f"{current.get('temperature_2m','?')} {unit_temp}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("체감 온도", f"{current.get('apparent_temperature','?')} {unit_temp}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("습도", f"{current.get('relative_humidity_2m','?')} %")
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("풍속", f"{current.get('wind_speed_10m','?')} km/h")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.markdown("##### 오늘/내일 요약", unsafe_allow_html=True)
    if not df_daily.empty:
        cc = st.columns(min(5, len(df_daily)))
        for i in range(min(len(df_daily), 5)):
            row = df_daily.iloc[i]
            txt, emo = code_to_text_emoji(row.get("weathercode"))
            with cc[i]:
                st.markdown(
                    f"<div class='glass metric-card {section_anim}'>"
                    f"<b>{row['time'].strftime('%m/%d (%a)')}</b><br>"
                    f"{emo} {txt}<br>"
                    f"최고 {row['temperature_2m_max']}{unit_temp} · 최저 {row['temperature_2m_min']}{unit_temp}<br>"
                    f"강수 {row.get('precipitation_sum', 0)} mm"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------
# 시간별
# ------------------------------------------------------------
elif choice == "시간별":
    st.subheader("⏱️ 시간별 예보")
    if view_hourly.empty:
        st.info("표시할 시간별 데이터가 없습니다.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view_hourly["time"], y=view_hourly["temperature_2m"],
                                 mode="lines+markers", name=f"기온({unit_temp})"))
        if "precipitation_probability" in view_hourly.columns:
            fig.add_trace(go.Bar(x=view_hourly["time"], y=view_hourly["precipitation_probability"],
                                 name="강수확률(%)", opacity=0.4, yaxis="y2"))
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="시간",
            yaxis_title=f"기온({unit_temp})",
            yaxis2=dict(title="강수확률(%)", overlaying="y", side="right", range=[0, 100]),
            legend=dict(orientation="h"),
            hovermode="x unified",
            transition_duration=300  # Plotly 애니메이션 전환
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("데이터 보기", expanded=False):
            st.dataframe(make_hourly_display(view_hourly, unit_temp), use_container_width=True, height=280)

# ------------------------------------------------------------
# 일별
# ------------------------------------------------------------
elif choice == "일별":
    st.subheader("📅 일별 예보")
    if df_daily.empty:
        st.info("표시할 일별 데이터가 없습니다.")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_daily["time"], y=df_daily["temperature_2m_max"],
                                  mode="lines+markers", name=f"최고기온({unit_temp})", line=dict(width=2)))
        fig2.add_trace(go.Scatter(x=df_daily["time"], y=df_daily["temperature_2m_min"],
                                  mode="lines+markers", name=f"최저기온({unit_temp})", line=dict(width=2, dash="dot")))
        if "precipitation_probability_max" in df_daily.columns:
            fig2.add_trace(go.Bar(x=df_daily["time"], y=df_daily["precipitation_probability_max"],
                                  name="최대 강수확률(%)", opacity=0.35, yaxis="y2"))
        fig2.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="날짜",
            yaxis_title=f"기온({unit_temp})",
            yaxis2=dict(title="강수확률(%)", overlaying="y", side="right", range=[0, 100]),
            legend=dict(orientation="h"),
            hovermode="x unified",
            transition_duration=300
        )
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("데이터 보기", expanded=False):
            st.dataframe(make_daily_display(df_daily, unit_temp), use_container_width=True, height=280)

# ------------------------------------------------------------
# 지도
# ------------------------------------------------------------
else:
    st.subheader("🗺️ 지도")
    with st.expander("지도 보기 (Leaflet)", expanded=True):
        fmap = folium.Map(location=[selected_lat, selected_lon], zoom_start=10, control_scale=True)
        folium.Marker(
            [selected_lat, selected_lon],
            popup=f"{selected_name}",
            tooltip="선택 위치",
            icon=folium.Icon(color="blue", icon="cloud"),
        ).add_to(fmap)
        st_folium(fmap, height=360, use_container_width=True)

# ------------------------------------------------------------
# 푸터
# ------------------------------------------------------------

