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
try:
    # Kaggle is optional; only import if installed
    from kaggle import api as kaggle_api  # type: ignore
    KAGGLE_AVAILABLE = True
except Exception:
    kaggle_api = None  # type: ignore
    KAGGLE_AVAILABLE = False


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
    # 전역 템플릿 설정은 환경에 따라 제약이 있어, 각 figure 수준에서 update_layout(font=...)를 사용합니다.
    return


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
# Kaggle 유틸
# -----------------------------
def kaggle_authenticate_if_possible() -> bool:
    if not KAGGLE_AVAILABLE:
        return False
    # Prefer config dir created by app or repo Kaggle.json
    config_dir = os.path.join(os.getcwd(), ".kaggle")
    repo_kaggle = os.path.join(os.getcwd(), "Kaggle.json")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "kaggle.json")
    if os.path.exists(repo_kaggle) and not os.path.exists(config_path):
        try:
            with open(repo_kaggle, "rb") as src, open(config_path, "wb") as dst:
                dst.write(src.read())
            try:
                os.chmod(config_path, 0o600)
            except Exception:
                pass
            os.environ["KAGGLE_CONFIG_DIR"] = config_dir
        except Exception:
            pass
    # Try to authenticate
    try:
        kaggle_api.authenticate()  # type: ignore[attr-defined]
        return True
    except Exception:
        return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))


def ensure_kaggle_file(dataset_slug: str, filename: str) -> str:
    if not KAGGLE_AVAILABLE:
        raise RuntimeError("Kaggle 라이브러리가 설치되어 있지 않습니다.")
    if not kaggle_authenticate_if_possible():
        raise RuntimeError("Kaggle 인증이 필요합니다. Kaggle 탭에서 kaggle.json을 업로드하거나 환경변수를 설정하세요.")
    dl_dir = os.path.join(os.getcwd(), "kaggle_data")
    os.makedirs(dl_dir, exist_ok=True)
    target_path = os.path.join(dl_dir, filename)
    if not os.path.exists(target_path):
        kaggle_api.dataset_download_file(dataset=dataset_slug, file_name=filename, path=dl_dir, force=False, quiet=True)  # type: ignore[union-attr]
        # download_file leaves as .csv or .zip? For single file, it downloads exact file; if zipped, handle
        if not os.path.exists(target_path):
            # Try full dataset download & unzip as fallback
            kaggle_api.dataset_download_files(dataset_slug, path=dl_dir, unzip=True, quiet=True)  # type: ignore[union-attr]
    if not os.path.exists(target_path):
        # After unzip, search for the file
        for f in os.listdir(dl_dir):
            if f.lower() == filename.lower():
                target_path = os.path.join(dl_dir, f)
                break
    if not os.path.exists(target_path):
        raise RuntimeError(f"Kaggle 파일을 찾을 수 없습니다: {dataset_slug} / {filename}")
    return target_path


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
    """
    Kaggle의 Berkeley Earth Surface Temperatures에서 국가별 월평균기온을 불러와
    연도별 평균(절대값, °C)으로 집계합니다.
    데이터셋: berkeleyearth/climate-change-earth-surface-temperature-data
    파일: GlobalLandTemperaturesByCountry.csv
    반환 컬럼: entity(국가명), year, date(연말), value(해당 연도 평균기온 °C)
    """
    dataset = "berkeleyearth/climate-change-earth-surface-temperature-data"
    filename = "GlobalLandTemperaturesByCountry.csv"
    path = ensure_kaggle_file(dataset, filename)
    raw = pd.read_csv(path)
    # 컬럼: dt, AverageTemperature, AverageTemperatureUncertainty, Country
    need = [c for c in ["dt", "AverageTemperature", "Country"] if c in raw.columns]
    if len(need) < 3:
        raise RuntimeError("예상 컬럼(dt, AverageTemperature, Country)이 없습니다.")
    df = raw[need].rename(columns={"dt": "date", "AverageTemperature": "temp", "Country": "entity"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "temp", "entity"]).copy()
    df["year"] = df["date"].dt.year
    # 연도별 평균기온으로 집계
    ann = (
        df.groupby(["entity", "year"], as_index=False)
        .agg(value=("temp", "mean"))
    )
    ann["date"] = pd.to_datetime(ann["year"].astype(int).astype(str) + "-12-31")
    ann = ann[["entity", "year", "date", "value"]].sort_values(["year", "entity"]).reset_index(drop=True)
    return ann


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
        locations="entity",
        locationmode="country names",
        color="value",
        hover_name="entity",
        color_continuous_scale="RdYlBu_r",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar=dict(title="평균기온(°C)"))
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

# 사이드바(공통 옵션)
st.sidebar.header("옵션")
smooth_co2em = st.sidebar.number_input("연간 CO₂ 배출량: 스무딩(이동평균)", min_value=0, max_value=60, value=0, step=1)
smooth_noaa = st.sidebar.number_input("Mauna Loa CO₂: 스무딩(이동평균)", min_value=0, max_value=24, value=6, step=1)
smooth_gis = st.sidebar.number_input("기온 이상: 스무딩(이동평균)", min_value=0, max_value=24, value=12, step=1)

# KPI 영역
try:
    _em = load_owid_co2_emissions_world()
    em_last = _em.dropna().sort_values("date").iloc[-1]["value"] if len(_em) else None
except Exception:
    em_last = None
try:
    _co2 = load_noaa_mlo_co2_monthly()
    co2_last = _co2.dropna().sort_values("date").iloc[-1]["value"] if len(_co2) else None
except Exception:
    co2_last = None
try:
    _gt = load_nasa_gistemp_monthly_global()
    gt_last = _gt.dropna().sort_values("date").iloc[-1]["value"] if len(_gt) else None
except Exception:
    gt_last = None

col_a, col_b, col_c = st.columns(3)
col_a.metric("전세계 CO₂ 배출량(최근, Mt)", f"{em_last:,.0f}" if em_last is not None else "-")
col_b.metric("대기 CO₂(최근, ppm)", f"{co2_last:,.2f}" if co2_last is not None else "-")
col_c.metric("전지구 이상기온(최근, °C)", f"{gt_last:+.2f}" if gt_last is not None else "-")

st.markdown("---")

# 탭 구성
extra_tabs = ["Kaggle"] if KAGGLE_AVAILABLE else []
tab1, tab2, tab3, tab4, *rest = st.tabs(["시계열", "세계 지도", "상관관계", "사용자 설명", *extra_tabs])

with tab1:
    st.subheader("시계열 지표")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 전세계 CO₂ 배출량 (연간, Mt)")
        try:
            df_em = load_owid_co2_emissions_world()
        except Exception:
            st.warning("공식 데이터 로드 실패 → 예시 데이터로 표시합니다.")
            df_em = sample_world_emissions()
        df_em = df_em.dropna().drop_duplicates()
        fig_em = line_chart(df_em, "전세계 CO₂ 배출량", "배출량 (Mt)", smooth=smooth_co2em)
        st.plotly_chart(fig_em, use_container_width=True)
        download_button_for_df(df_em, "CSV 다운로드(전세계 CO₂ 배출량)", "world_co2_emissions.csv")

    with c2:
        st.markdown("#### 대기 중 CO₂ (월별, ppm)")
        try:
            df_co2 = load_noaa_mlo_co2_monthly()
        except Exception:
            st.warning("NOAA CO₂ 로드 실패 → 예시 데이터로 표시합니다.")
            df_co2 = sample_noaa_co2()
        df_co2 = df_co2.dropna().drop_duplicates()
        fig_co2 = line_chart(df_co2, "Mauna Loa 대기 CO₂", "농도 (ppm)", smooth=smooth_noaa)
        st.plotly_chart(fig_co2, use_container_width=True)
        download_button_for_df(df_co2, "CSV 다운로드(대기 CO₂)", "noaa_co2_monthly.csv")

    st.markdown("#### 전지구 평균기온 이상(월별, °C)")
    try:
        df_temp = load_nasa_gistemp_monthly_global()
    except Exception:
        st.warning("NASA GISTEMP 로드 실패 → 예시 데이터로 표시합니다.")
        df_temp = sample_gistemp()
    df_temp = df_temp.dropna().drop_duplicates()
    fig_temp = line_chart(df_temp, "전지구 평균기온 이상(월별)", "이상기온 (°C)", smooth=smooth_gis)
    st.plotly_chart(fig_temp, use_container_width=True)
    download_button_for_df(df_temp, "CSV 다운로드(전지구 이상기온)", "nasa_gistemp_global_monthly.csv")

with tab2:
    st.subheader("국가별 평균기온 세계 지도(연도별)")
    try:
        df_country = load_country_temperature_change()
    except Exception as e:
        st.error(f"국가별 온도 데이터 로드 실패: {e}")
        df_country = None
    if df_country is not None and len(df_country):
        df_country = df_country.copy()
        df_country["date"] = pd.to_datetime(df_country["date"])  # 표준화
        years = sorted(df_country["date"].dt.year.unique().tolist())
        if years:
            sel_year = st.slider("연도 선택", min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)), step=1)
            fig_map = choropleth_by_year(df_country, sel_year, f"국가별 평균기온(°C) - {sel_year}")
            st.plotly_chart(fig_map, use_container_width=True)
            dl_df = df_country.rename(columns={"entity": "group"})[["date", "value", "group"]].sort_values(["date", "group"])
            download_button_for_df(dl_df, "CSV 다운로드(국가별 평균기온)", "country_avg_temperature_annual.csv")
        else:
            st.info("표시할 연도가 없습니다.")
        st.caption("참고: 일부 국가/연도는 값이 없을 수 있습니다.")
    else:
        st.info("실제 데이터를 사용하려면 Kaggle 인증이 필요합니다. Kaggle 탭에서 kaggle.json 업로드 또는 환경변수를 설정하세요. Docker 실행을 권장합니다.")

with tab3:
    st.subheader("기온과 학업 성취 상관관계 (실제 데이터)")
    st.markdown("""
    1) Kaggle 탭에서 학업 성취(예: PISA) 관련 CSV를 다운로드하세요.
    2) 아래에서 CSV 파일을 선택하고, 국가/연도/성취도(숫자) 열을 지정합니다.
    3) 동일 연도의 국가별 평균기온(베를리 지구 Kaggle 데이터)과 병합하여 산점도 및 상관계수를 보여줍니다.
    """)
    # 온도 데이터 준비
    try:
        df_temp_c = load_country_temperature_change()
    except Exception as e:
        # Kaggle 미설치 시에는 명확한 에러 문구로 안내
        if not KAGGLE_AVAILABLE:
            st.error("국가별 온도 데이터 로드 실패: Kaggle 라이브러리가 설치되어 있지 않습니다.")
        else:
            st.error(f"온도 데이터 로드 실패: {e}")
            st.info("Kaggle 인증이 필요할 수 있습니다. Kaggle 탭에서 kaggle.json 업로드 또는 환경변수를 설정하세요.")
        df_temp_c = None

    if df_temp_c is not None and len(df_temp_c):
        dl_dir = os.path.join(os.getcwd(), "kaggle_data")
        csv_files = [f for f in os.listdir(dl_dir)] if os.path.isdir(dl_dir) else []
        csv_files = [f for f in csv_files if f.lower().endswith(".csv")]
        up_alt = st.file_uploader("(대안) 교육 성취 CSV 직접 업로드", type=["csv"], accept_multiple_files=False)
        if not csv_files and up_alt is None:
            st.info("kaggle_data 폴더에 CSV가 없습니다. Kaggle 탭에서 먼저 다운로드하거나, 위에 CSV를 업로드하세요.")
        else:
        df_edu_raw = None
        if up_alt is not None:
            try:
                df_edu_raw = pd.read_csv(up_alt)
            except Exception as e:
                st.error(f"업로드 CSV 읽기 실패: {e}")
        else:
            sel_file = st.selectbox("학업 성취 CSV 선택", csv_files)
            edu_path = os.path.join(dl_dir, sel_file)
            try:
                df_edu_raw = pd.read_csv(edu_path)
            except Exception as e:
                st.error(f"CSV 읽기 실패: {e}")
                df_edu_raw = None
        if df_edu_raw is not None:
            st.write("행/열:", df_edu_raw.shape)
            st.dataframe(df_edu_raw.head(50))
            cols = df_edu_raw.columns.tolist()
            col_country = st.selectbox("국가 열", cols)
            col_year = st.selectbox("연도 열", cols)
            col_score = st.selectbox("성취도(숫자) 열", cols)

            # 전처리
            edu = df_edu_raw[[col_country, col_year, col_score]].copy()
            edu.columns = ["entity", "year", "score"]
            # 연도 숫자화
            edu["year"] = pd.to_numeric(edu["year"], errors="coerce").astype("Int64")
            # 점수 숫자화
            edu["score"] = pd.to_numeric(edu["score"], errors="coerce")
            edu = edu.dropna(subset=["entity", "year", "score"]).copy()

            # 연도 범위 선택
            if df_temp_c is not None and len(df_temp_c) and len(edu):
                y_min = int(max(edu["year"].min(), df_temp_c["year"].min()))
                y_max = int(min(edu["year"].max(), df_temp_c["year"].max()))
            else:
                y_min, y_max = 2000, 2018
            if y_min > y_max:
                y_min, y_max = y_max, y_max
            year_sel = st.slider("연도 선택(상관계수 계산 연도)", min_value=y_min, max_value=y_max, value=y_max, step=1)

            if df_temp_c is None:
                st.warning("온도 데이터가 없어 상관관계를 계산할 수 없습니다. Kaggle 인증 후 다시 시도하세요.")
            else:
                # 동일 연도 병합(국가명 기준)
                temp_y = df_temp_c[df_temp_c["year"] == year_sel][["entity", "value"]].rename(columns={"value": "temp"})
                edu_y = edu[edu["year"] == year_sel][["entity", "score"]]
                merged = pd.merge(temp_y, edu_y, on="entity", how="inner")
                st.write(f"병합된 국가 수: {len(merged)}")
                if len(merged) < 5:
                    st.warning("충분한 국가 수가 없습니다. 다른 연도를 선택하거나 매핑을 조정해 보세요.")
                else:
                    # 피어슨 상관계수
                    try:
                        r = float(np.corrcoef(merged["temp"], merged["score"])[0, 1])
                    except Exception:
                        r = float("nan")
                    st.metric("피어슨 상관계수 r", f"{r:.3f}" if pd.notna(r) else "NaN")
                    # 산점도 + 단순 선형회귀선
                    fig_sc = px.scatter(merged, x="temp", y="score", hover_name="entity", title=f"{year_sel}년: 평균기온(°C) vs 학업 성취")
                    # 회귀선 수동 추가
                    try:
                        m, b = np.polyfit(merged["temp"].astype(float), merged["score"].astype(float), 1)
                        xfit = np.linspace(float(merged["temp"].min()), float(merged["temp"].max()), 50)
                        yfit = m * xfit + b
                        fig_sc.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="회귀선"))
                    except Exception:
                        pass
                    fig_sc.update_layout(xaxis_title="평균기온(°C)", yaxis_title="학업 성취(점수)")
                    st.plotly_chart(fig_sc, use_container_width=True)

with tab4:
    st.subheader("사용자 입력 대시보드 (설명 기반)")
    with st.expander("설명 전문 보기", expanded=False):
        st.write(DESCRIPTION_TEXT)

    df_desc, df_desc_std = build_description_based_dataset()
    metrics = ["전체"] + sorted(df_desc["지표"].unique().tolist())
    sel_metric = st.selectbox("지표 필터", metrics, index=0)
    show_sankey = st.checkbox("관계 흐름(상키) 보기", value=True)

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

############################################
# Kaggle 탭 (선택, 라이브러리 설치 시 표시)
############################################
if KAGGLE_AVAILABLE and rest:
    with rest[0]:
        st.subheader("Kaggle 데이터셋 다운로드 및 미리보기")
        st.caption("주의: Kaggle 계정의 API 토큰이 필요합니다. 이 앱은 업로드된 kaggle.json을 세션 임시 폴더로 설정하여 인증합니다.")

        # 인증 준비: 파일 업로드 또는 기존 Kaggle.json 활용
        up = st.file_uploader("kaggle.json 업로드", type=["json"], accept_multiple_files=False)
        kaggle_info = None
        config_dir = os.path.join(os.getcwd(), ".kaggle")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "kaggle.json")

        if up is not None:
            try:
                content = up.read()
                with open(config_path, "wb") as f:
                    f.write(content)
                try:
                    os.chmod(config_path, 0o600)
                except Exception:
                    pass
                os.environ["KAGGLE_CONFIG_DIR"] = config_dir
                kaggle_info = "업로드한 kaggle.json으로 인증 설정 완료"
            except Exception as e:
                st.error(f"kaggle.json 저장 실패: {e}")

        # 워크스페이스 루트에 Kaggle.json이 존재하면 사용
        if kaggle_info is None:
            repo_kaggle = os.path.join(os.getcwd(), "Kaggle.json")
            if os.path.exists(repo_kaggle):
                try:
                    with open(repo_kaggle, "rb") as src, open(config_path, "wb") as dst:
                        dst.write(src.read())
                    try:
                        os.chmod(config_path, 0o600)
                    except Exception:
                        pass
                    os.environ["KAGGLE_CONFIG_DIR"] = config_dir
                    kaggle_info = "리포지토리의 Kaggle.json으로 인증 설정 완료"
                except Exception as e:
                    st.warning(f"리포지토리 Kaggle.json 사용 실패: {e}")

        # 환경변수 인증이 이미 있는 경우 안내
        if kaggle_info is None and os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
            kaggle_info = "환경변수(KAGGLE_USERNAME/KEY)로 인증 사용"

        if kaggle_info:
            st.success(kaggle_info)
        else:
            st.info("kaggle.json 업로드 또는 환경변수 설정이 필요합니다.")

        dataset_slug = st.text_input("데이터셋 슬러그 (owner/dataset)", value="zynicide/wine-reviews")
        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            if st.button("파일 목록 조회"):
                try:
                    # Top-level optional import already set KAGGLE_AVAILABLE and kaggle_api
                    try:
                        kaggle_api.authenticate()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    files = kaggle_api.dataset_list_files(dataset_slug)  # type: ignore[union-attr]
                    st.write("파일 목록:", [f.name for f in files.files])
                except Exception as e:
                    st.error(f"파일 목록 조회 실패: {e}")
        with col_dl2:
            if st.button("데이터셋 다운로드"):
                try:
                    # 다운로드 디렉토리
                    dl_dir = os.path.join(os.getcwd(), "kaggle_data")
                    os.makedirs(dl_dir, exist_ok=True)
                    try:
                        kaggle_api.authenticate()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    kaggle_api.dataset_download_files(dataset_slug, path=dl_dir, unzip=True, quiet=False)  # type: ignore[union-attr]
                    st.success(f"다운로드 완료: {dl_dir}")
                except Exception as e:
                    st.error(f"다운로드 실패: {e}")

        # CSV 미리보기
        dl_dir = os.path.join(os.getcwd(), "kaggle_data")
        if os.path.isdir(dl_dir):
            csv_files = [f for f in os.listdir(dl_dir) if f.lower().endswith(".csv")]
            if csv_files:
                sel_csv = st.selectbox("CSV 파일 선택", csv_files)
                if sel_csv:
                    try:
                        df_k = pd.read_csv(os.path.join(dl_dir, sel_csv))
                        st.write("행/열:", df_k.shape)
                        st.dataframe(df_k.head(200))

                        # 표준화 매핑
                        st.markdown("#### 표준 스키마 매핑 (date/value/(선택)group)")
                        cols = df_k.columns.tolist()
                        date_col = st.selectbox("날짜 열", ["(없음)"] + cols, index=0)
                        value_col = st.selectbox("값 열", cols, index=min(1, len(cols)-1))
                        group_col = st.selectbox("그룹 열(선택)", ["(없음)"] + cols, index=0)

                        def to_datetime_safe(s: pd.Series) -> pd.Series:
                            try:
                                return pd.to_datetime(s, errors="coerce")
                            except Exception:
                                return pd.to_datetime(pd.Series([None]*len(s)), errors="coerce")

                        std = pd.DataFrame()
                        if date_col != "(없음)":
                            std["date"] = to_datetime_safe(df_k[date_col])
                        else:
                            std["date"] = pd.NaT
                        std["value"] = pd.to_numeric(df_k[value_col], errors="coerce")
                        if group_col != "(없음)":
                            std["group"] = df_k[group_col].astype(str)
                        else:
                            std["group"] = "all"
                        std = std.dropna(subset=["value"]).copy()

                        st.markdown("#### 표준화 데이터 미리보기")
                        st.dataframe(std.head(200))
                        download_button_for_df(std, "CSV 다운로드(표준화)", f"kaggle_standardized_{sel_csv}")

                        # 간단 시각화
                        if std["date"].notna().any():
                            st.markdown("#### 빠른 선그래프")
                            fig_k = px.line(std.sort_values("date"), x="date", y="value", color="group")
                            fig_k.update_layout(xaxis_title="날짜", yaxis_title="값")
                            st.plotly_chart(fig_k, use_container_width=True)
                    except Exception as e:
                        st.error(f"CSV 미리보기 실패: {e}")
            else:
                st.info("kaggle_data 폴더에 CSV 파일이 없습니다. 먼저 다운로드하세요.")
        else:
            st.info("아직 다운로드 폴더가 없습니다. 먼저 데이터셋을 다운로드하세요.")

