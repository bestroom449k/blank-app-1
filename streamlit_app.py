from __future__ import annotations

import os
import io
import base64
from datetime import datetime, timezone, date
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# 폰트 적용 시도 (없으면 자동 생략)
# -----------------------------
def apply_pretendard_font():
    candidates = [
        os.path.join("fonts", "Pretendard-Bold.ttf"),
        os.path.join(os.sep, "fonts", "Pretendard-Bold.ttf"),
        "Pretendard-Bold.ttf",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    if font_path:
        try:
            with open(font_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f"""
                <style>
                @font-face {{
                  font-family: 'Pretendard';
                  src: url(data:font/ttf;base64,{b64}) format('truetype');
                  font-weight: 700;
                  font-style: normal;
                }}
                html, body, [class*="css"] {{
                  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', 'Apple SD Gothic Neo', '맑은 고딕', 'Malgun Gothic', 'Nanum Gothic', sans-serif;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    # Plotly 전역 폰트 설정(브라우저 폰트 우선)
    try:
        import plotly.io as pio  # type: ignore
        template = pio.templates["plotly"]
        template.layout.font.family = "Pretendard, Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"
        pio.templates.default = template
    except Exception:
        pass


# -----------------------------
# 공통 유틸
# -----------------------------
st.set_page_config(page_title="기후·학습 공개 데이터 & 사용자 설명 대시보드", layout="wide")
apply_pretendard_font()


def today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def drop_future(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    today = today_utc_date()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.date
    return out[out[date_col] <= today]


@st.cache_data(ttl=60 * 60)
def robust_get(url: str, params: Optional[dict] = None, retry: int = 3, timeout: int = 30) -> requests.Response:
    last_err: Optional[Exception] = None
    for _ in range(retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"요청 실패: {url} ({last_err})")


# -----------------------------
# 공개 데이터 로더
# -----------------------------
@st.cache_data(ttl=60 * 60)
def load_owid_co2_emissions_world() -> pd.DataFrame:
    # OWID 전세계 CO₂ 배출량 (Mt) 연도별
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    resp = robust_get(url)
    df = pd.read_csv(io.StringIO(resp.text))
    df = df[df["iso_code"] == "OWID_WRL"]
    slim = df[["year", "co2"]].rename(columns={"year": "year", "co2": "value"}).dropna()
    slim["date"] = pd.to_datetime(slim["year"].astype(int).astype(str) + "-12-31")
    out = slim[["date", "value"]].sort_values("date")
    out = drop_future(out, "date").reset_index(drop=True)
    return out.dropna().drop_duplicates()


@st.cache_data(ttl=60 * 60)
def load_noaa_mlo_co2_monthly() -> pd.DataFrame:
    # NOAA Mauna Loa CO₂ 월별 농도
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    resp = robust_get(url)
    df = pd.read_csv(io.StringIO(resp.text), comment="#")
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    year_col = cols_lower.get("year")
    month_col = cols_lower.get("month")
    trend_col = cols_lower.get("trend") or cols_lower.get("average")
    if not (year_col and month_col and trend_col):
        raise RuntimeError("예상 컬럼(year, month, trend/average) 없음")
    df["date"] = pd.to_datetime(
        df[year_col].astype(int).astype(str)
        + "-"
        + df[month_col].astype(int).astype(str).str.zfill(2)
        + "-15"
    )
    df["value"] = pd.to_numeric(df[trend_col], errors="coerce")
    out = df.dropna(subset=["value"])[["date", "value"]].sort_values("date")
    out = drop_future(out, "date").reset_index(drop=True)
    return out.dropna().drop_duplicates()


@st.cache_data(ttl=60 * 60)
def load_nasa_gistemp_monthly_global() -> pd.DataFrame:
    # NASA GISTEMP v4 전지구 월별 기온 이상(°C)
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    resp = robust_get(url)
    raw = pd.read_csv(io.StringIO(resp.text), skiprows=1)
    month_cols = [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] if m in raw.columns]
    if not month_cols or "Year" not in raw.columns:
        raise RuntimeError("NASA GISTEMP 스키마 변동")
    df = raw[["Year"] + month_cols].copy()
    df = df.melt(id_vars="Year", value_vars=month_cols, var_name="month", value_name="anomaly")
    df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce")
    scale = 0.01 if df["anomaly"].abs().mean(skipna=True) > 10 else 1.0
    df["anomaly"] = df["anomaly"] * scale
    month_map = dict(zip(month_cols, range(1, len(month_cols) + 1)))
    df["month_num"] = df["month"].map(month_map)
    df["date"] = pd.to_datetime(df["Year"].astype(int).astype(str) + "-" + df["month_num"].astype(int).astype(str) + "-15")
    out = df.dropna(subset=["anomaly"])[["date", "anomaly"]].rename(columns={"anomaly": "value"}).sort_values("date")
    out = drop_future(out, "date").reset_index(drop=True)
    return out.dropna().drop_duplicates()


@st.cache_data(ttl=60 * 60)
def load_country_temperature_change() -> pd.DataFrame:
    # Berkeley Earth 기반(OWID 가공본), 국가별 연도별 온도 변화(°C)
    url_candidates = [
        "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Temperature%20change%20-%20Berkeley%20Earth/Temperature%20change%20-%20Berkeley%20Earth.csv",
        "https://raw.githubusercontent.com/owid/owid-data/master/processed/berkeley_temperature_change/berkeley_temperature_change.csv",
        "https://raw.githubusercontent.com/owid/owid-data/master/processed/temperature_change/temperature_change.csv",
    ]
    df = None
    last_err: Optional[Exception] = None
    for u in url_candidates:
        try:
            resp = robust_get(u)
            _df = pd.read_csv(io.StringIO(resp.text))
            if not _df.empty:
                df = _df
                break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        raise RuntimeError(f"국가별 온도 변화 데이터 로드 실패: {last_err}")

    cols = [str(c) for c in df.columns]
    low = {c.lower().strip(): c for c in cols}
    col_entity = low.get("entity") or low.get("country") or low.get("location")
    col_code = low.get("code") or low.get("iso3") or low.get("iso_code") or low.get("iso")
    col_year = low.get("year")
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    pref = [c for c in numeric_cols if any(k in c.lower() for k in ["temp", "temperature", "change"])]
    value_col = pref[-1] if pref else (numeric_cols[-1] if numeric_cols else None)
    if not (col_entity and col_code and col_year and value_col):
        raise RuntimeError("국가/연도/값 컬럼 식별 실패")
    slim = df[[col_entity, col_code, col_year, value_col]].rename(
        columns={col_entity: "entity", col_code: "code", col_year: "year", value_col: "value"}
    )
    slim = slim.dropna(subset=["code", "year", "value"]).copy()
    slim["year"] = slim["year"].astype(int)
    slim["date"] = pd.to_datetime(slim["year"].astype(str) + "-12-31")
    out = slim[["code", "entity", "date", "value", "year"]].sort_values(["year", "code"]).reset_index(drop=True)
    return out


# -----------------------------
# 예시 데이터(공개 API 실패 시 대체)
# -----------------------------
def sample_world_emissions() -> pd.DataFrame:
    years = [2015, 2018, 2021, 2024]
    vals = [35000, 36500, 37000, 37500]
    return pd.DataFrame({"date": pd.to_datetime([f"{y}-12-31" for y in years]), "value": vals})


def sample_noaa_co2() -> pd.DataFrame:
    d = pd.date_range("2022-01-15", periods=12, freq="MS") + pd.Timedelta(days=14)
    v = np.linspace(415, 422, len(d))
    return pd.DataFrame({"date": d, "value": v})


def sample_gistemp() -> pd.DataFrame:
    d = pd.date_range("2021-01-15", periods=24, freq="MS") + pd.Timedelta(days=14)
    v = 0.6 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, len(d)))
    return pd.DataFrame({"date": d, "value": v})


def sample_country_temp_change() -> pd.DataFrame:
    data = [
        {"code": "KOR", "entity": "Korea, Republic of", "year": 2010, "value": 0.8},
        {"code": "KOR", "entity": "Korea, Republic of", "year": 2020, "value": 1.2},
        {"code": "USA", "entity": "United States", "year": 2010, "value": 0.9},
        {"code": "USA", "entity": "United States", "year": 2020, "value": 1.3},
        {"code": "JPN", "entity": "Japan", "year": 2010, "value": 0.7},
        {"code": "JPN", "entity": "Japan", "year": 2020, "value": 1.1},
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
    return df[["code", "entity", "date", "value", "year"]]


# -----------------------------
# 시각화 유틸
# -----------------------------
def line_chart(df: pd.DataFrame, title: str, y_title: str, smooth: int = 0) -> go.Figure:
    data = df.copy().sort_values("date").dropna()
    ycol = "value"
    if smooth and smooth > 1:
        data["value_smooth"] = data["value"].rolling(smooth, min_periods=1, center=True).mean()
        ycol = "value_smooth"
    fig = px.line(data, x="date", y=ycol, title=title)
    fig.update_layout(xaxis_title="날짜", yaxis_title=y_title)
    return fig


def choropleth_by_year(df: pd.DataFrame, year: int, title: str) -> go.Figure:
    snap = df[df["date"].dt.year == year]
    fig = px.choropleth(
        snap,
        locations="code",
        color="value",
        hover_name="entity",
        color_continuous_scale="RdYlBu_r",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar=dict(title="온도 변화(°C)"))
    return fig


def download_button_for_df(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=file_name, mime="text/csv")


# -----------------------------
# 사용자 설명 기반 데이터
# -----------------------------
DESCRIPTION_TEXT = """
서론 (문제 제기)
기후 변화가 계속 됨에 따라 학생들이 기온 변화로 인해 평소 보다 더 큰 스트레스를 받는 것을 느꼈다. 점점 높아지는 최고기온과 길어지는 더위가 학생들의 학업 성적에 전혀 무관할까? 하는 궁금증을 가진 우리는 학생들의 학업에 미치는 영향에 관한 궁금증을 해소하고, 이를 학생들에게 알려 기온 상승 상황에서 성적 상승을 돕기위해서 이 주제를 선정해 연구한다.

인류 활동으로 발생한 지구 온실가스 배출량은 점점 증가하고 대기 중 이산화탄소 농도가  높아지며 전지구 평균기온도 높아지고 있다. 화석 연료 사용 증가 및 산업 발전과 인구 증가로 인해 과도한 에너지 소비가 일어나는 것이 주된 원인이다.

본론 1 (데이터 분석)
이번 연구에서는 기후 변화와 수면 시간의 관계를 분석하였다. 최근 기온 상승은 단순한 생활 불편이 아닌 청소년들의 수면 패턴에도 큰 영향을 주고 있다. 선 그래프를 통해 분석 한 결과, 기온이 높아질수록 평균 수면시간이 점차 줄어드는 경향이 확인되었다.
특히 더운 날씨에는 학생들이 깊은 잠에 드는 시간이 짧아지고, 자주 깨는 경우가 많아 수면의 질 또한 떨어지는 모습을 보였다. 이는 곧 수면 부족으로 이어지며, 학습 효율 저하와 집중력 감소의 원인으로 작용할 수 있다.
따라서 기후 변화는 단순히 환경적 위기만이 아니라, 청소년들의 수면과 학습 능력에 영향을 주는 중요한 요인임을 알 수 있다.

본론 2 (원인 및 영향 탐구)
기온 변화와 성적은 실제로 상관관계를 가진다.
아래의 막대 그래프와 산점도를 살피면 더 정확히 알 수 있다.

위의 막대그래프는 전 세계 다양한 연구에서 보고된 기온 상승과 학업 성적 변화의 상관관계를 비교한 것이다. 그래프에서 볼 수 있듯이, 대부분의 연구에서 기온이 일정 수준 이상 상승하면 학생들의 성적이 표준편차 단위로 감소하는 경향이 나타난다.
특히 OECD 국제학업성취도 평가(PISA)를 활용한 58개국 분석에서는 고온 노출이 누적될수록 성적이 크게 하락하는 결과가 확인되었다. 미국과 뉴욕의 사례 또한 시험 당일 기온이 높을수록 학생들의 성적과 합격률이 유의하게 떨어졌다. 한국의 경우 단일 고온일의 효과는 비교적 작지만, 34℃ 이상의 날이 누적될수록 수학과 영어 성적이 점차 감소하는 경향을 보였다.
이처럼 고온 환경은 단순한 불쾌감을 넘어 학업 성취에도 부정적인 영향을 미치며, 특히 누적 효과가 장기적인 성적 저하로 이어질 수 있음을 시사한다.

위 그림은 2012년 PISA(패널 A) 또는 SEDA(패널 B) 수학 평균 점수와 국가 또는 미국 카운티별 연평균 기온의 산점도이다.

연평균 기온은 1980년부터 2011년까지 측정되었고, 패널 B는 평균 기온 분포의 백분위 별로 표준화된 3~8학년 수학 점수(2009~2013년)의 구간별 백분위 그래프이다. 점수는 과목, 학년, 연도별로 표준화된 점수를 사용한다.

이 산점도는 간단히 말하자면 미국 학생들의 수학 성적이 기온에 따라 어떻게 변화하는지 보여준다. 위 그래프는 기온이 높아질 수록 성적이 하락하는 경향을 면밀히 보여주고 있다. 따라서 우리는 기온 상승이 학생들의 성적에 밀접한 연관을 가진다는 사실을 알 수 있다.

결론 (제언)
이번 연구를 통해 우리는 기온 상승이 단순히 생활 불편에 그치지 않고, 학생들의 수면 질 저하와 집중력 감소를 초래하며, 학업 성취도에 부정적인 영향을 줄 수 있음을 확인하였다. 특히 기온이 일정 수준 이상 상승할 경우 성적이 표준편차 단위로 감소하는 경향이 여러 연구에서 공통적으로 드러났다. 이는 단일 요인이 아닌, 반복적이고 누적된 높은 온도 노출이 장기적으로 학생들의 학습 능력을 저해한다는 점을 보여준다.
"""


def build_description_based_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [
        {"연구": "PISA 58개국", "지표": "성적", "방향": "하락", "강도": "높음"},
        {"연구": "미국(전국)", "지표": "성적", "방향": "하락", "강도": "보통"},
        {"연구": "뉴욕", "지표": "성적", "방향": "하락", "강도": "보통"},
        {"연구": "한국", "지표": "성적", "방향": "하락", "강도": "낮음"},
        {"연구": "전반", "지표": "수면시간", "방향": "감소", "강도": "보통"},
        {"연구": "전반", "지표": "수면질", "방향": "저하", "강도": "보통"},
    ]
    df = pd.DataFrame(records)
    order = {"낮음": 1, "보통": 2, "높음": 3}
    df["강도점수"] = df["강도"].map(order)
    df_std = df.copy()
    df_std["date"] = pd.NaT
    df_std["value"] = df_std["강도점수"]
    df_std["group"] = df_std["지표"]
    return df, df_std[["date", "value", "group"]]


# -----------------------------
# 앱 본문
# -----------------------------
st.title("기후 변화와 학습 영향 대시보드")

st.markdown(
    """
- 인류 활동에 따른 온실가스 배출 증가 → 대기 중 CO₂ 농도 상승 → 전지구 평균기온 상승
- 공개 데이터 기반 시계열과, 제공된 설명 기반 정성 시각화를 함께 제공합니다.
"""
)

st.header("공개 데이터 대시보드")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("전세계 CO₂ 배출량 (연간, Mt)")
    smooth_co2em = st.sidebar.number_input("연간 CO₂ 배출량: 스무딩(이동평균)", min_value=0, max_value=60, value=0, step=1)
    try:
        df_em = load_owid_co2_emissions_world()
    except Exception:
        st.warning("공식 데이터 로드 실패 → 예시 데이터로 표시합니다.")
        df_em = sample_world_emissions()
    df_em = df_em.dropna().drop_duplicates()
    fig_em = line_chart(df_em, "전세계 CO₂ 배출량", "배출량 (Mt)", smooth=smooth_co2em)
    st.plotly_chart(fig_em, use_container_width=True)
    download_button_for_df(df_em, "CSV 다운로드(전세계 CO₂ 배출량)", "world_co2_emissions.csv")

with col2:
    st.subheader("대기 중 CO₂ (월별, ppm)")
    smooth_noaa = st.sidebar.number_input("Mauna Loa CO₂: 스무딩(이동평균)", min_value=0, max_value=24, value=6, step=1)
    try:
        df_co2 = load_noaa_mlo_co2_monthly()
    except Exception:
        st.warning("NOAA CO₂ 로드 실패 → 예시 데이터로 표시합니다.")
        df_co2 = sample_noaa_co2()
    df_co2 = df_co2.dropna().drop_duplicates()
    fig_co2 = line_chart(df_co2, "Mauna Loa 대기 CO₂", "농도 (ppm)", smooth=smooth_noaa)
    st.plotly_chart(fig_co2, use_container_width=True)
    download_button_for_df(df_co2, "CSV 다운로드(대기 CO₂)", "noaa_co2_monthly.csv")

st.subheader("전지구 평균기온 이상(월별, °C)")
smooth_gis = st.sidebar.number_input("기온 이상: 스무딩(이동평균)", min_value=0, max_value=24, value=12, step=1)
try:
    df_temp = load_nasa_gistemp_monthly_global()
except Exception:
    st.warning("NASA GISTEMP 로드 실패 → 예시 데이터로 표시합니다.")
    df_temp = sample_gistemp()
df_temp = df_temp.dropna().drop_duplicates()
fig_temp = line_chart(df_temp, "전지구 평균기온 이상(월별)", "이상기온 (°C)", smooth=smooth_gis)
st.plotly_chart(fig_temp, use_container_width=True)
download_button_for_df(df_temp, "CSV 다운로드(전지구 이상기온)", "nasa_gistemp_global_monthly.csv")

st.markdown("---")

st.subheader("국가별 온도 변화 세계 지도(연도 선택)")
try:
    df_country = load_country_temperature_change()
except Exception:
    st.warning("국가별 온도 변화 데이터 로드 실패 → 예시 데이터로 표시합니다.")
    df_country = sample_country_temp_change()

df_country = df_country.copy()
df_country["date"] = pd.to_datetime(df_country["date"])  # 표준화
years = sorted(df_country["date"].dt.year.unique().tolist())
if years:
    sel_year = st.slider("연도 선택", min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)), step=1)
    fig_map = choropleth_by_year(df_country, sel_year, f"국가별 온도 변화(°C) - {sel_year}")
    st.plotly_chart(fig_map, use_container_width=True)
    dl_df = df_country.rename(columns={"code": "group"})[["date", "value", "group"]].sort_values(["date", "group"])
    download_button_for_df(dl_df, "CSV 다운로드(국가별 온도 변화)", "country_temperature_change.csv")
else:
    st.info("표시할 연도가 없습니다.")

st.caption("참고: 일부 국가/연도는 값이 없을 수 있습니다.")

st.markdown("---")

st.header("사용자 입력 대시보드 (설명 기반)")
with st.expander("설명 전문 보기", expanded=False):
    st.write(DESCRIPTION_TEXT)

df_desc, df_desc_std = build_description_based_dataset()

metrics = ["전체"] + sorted(df_desc["지표"].unique().tolist())
sel_metric = st.sidebar.selectbox("사용자 설명: 지표 필터", metrics, index=0)
show_sankey = st.sidebar.checkbox("관계 흐름(샌키) 보기", value=True)

if sel_metric != "전체":
    view_df = df_desc[df_desc["지표"] == sel_metric].copy()
else:
    view_df = df_desc.copy()

fig_bar = px.bar(
    view_df.sort_values("강도점수"),
    x="강도점수",
    y="연구",
    color="지표",
    orientation="h",
    title="설명 기반 정성적 강도(낮음=1, 보통=2, 높음=3)",
    text="강도",
)
fig_bar.update_layout(xaxis_title="강도점수(정성)", yaxis_title="연구")
st.plotly_chart(fig_bar, use_container_width=True)

if show_sankey:
    labels = ["기온 상승", "수면시간 감소", "수면질 저하", "학습 효율 저하", "성적 하락"]
    idx = {l: i for i, l in enumerate(labels)}
    sources = [idx["기온 상승"], idx["기온 상승"], idx["수면시간 감소"], idx["수면질 저하"], idx["학습 효율 저하"]]
    targets = [idx["수면시간 감소"], idx["수면질 저하"], idx["학습 효율 저하"], idx["학습 효율 저하"], idx["성적 하락"]]
    values = [1, 1, 1, 1, 1]
    sankey_fig = go.Figure(
        data=[go.Sankey(node=dict(label=labels, pad=15, thickness=20), link=dict(source=sources, target=targets, value=values))]
    )
    sankey_fig.update_layout(title="설명 기반 영향 흐름")
    st.plotly_chart(sankey_fig, use_container_width=True)

st.subheader("설명 기반 표준화 데이터 다운로드")
download_button_for_df(df_desc_std, "CSV 다운로드(설명 기반 표준화)", "description_based_standardized.csv")

st.markdown(
    """
###### 데이터 출처(공식):
- NOAA CO₂: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv
- NASA GISTEMP: https://data.giss.nasa.gov/gistemp/
- OWID CO₂(전세계): https://github.com/owid/co2-data
- 온도 변화(국가별, Berkeley Earth 정리): https://berkeleyearth.org/ (OWID 가공본)
"""
)

# Kaggle API는 본 앱에서 사용하지 않습니다. 필요 시 kaggle.json 설정 및 kaggle CLI로 인증 후 사용하세요.
