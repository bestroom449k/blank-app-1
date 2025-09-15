# streamlit_app.py
# -*- coding: utf-8 -*-
# ==========================================
# 주제: 기온 상승과 학업 성취(수면·성적) 연관 탐구 대시보드
# 실행 환경: Streamlit + GitHub Codespaces (또는 로컬)
# ------------------------------------------
# 공식 공개 데이터(코드로 직접 연결):
# - NASA GISTEMP v4 (전지구 월별 기온 이상, CSV)
#   페이지: https://data.giss.nasa.gov/gistemp/
#   CSV:    https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
#
# - World Bank EdStats Indicators API (PISA 지표 예: LO.PISA.MAT / LO.PISA.REA / LO.PISA.SCI)
#   문서: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
#   데이터탐색: https://databank.worldbank.org/source/education-statistics-%5E-all-indicators
#
# 추가 참고 자료(설명/문헌용, 본 앱은 직접 API 호출하지 않음):
# - 기후변화의 원인(기후위키, KMA): http://www.climate.go.kr/home/10_wiki/index.php/%EA%B8%B0%ED%9B%84%EB%B3%80%ED%99%94%EC%9D%98_%EC%9B%90%EC%9D%B8
# - 한겨레: 기후변화가 연간 44시간의 수면을 앗아갔다: https://www.hani.co.kr/arti/science/science_general/1046883.html
# - IPCC AR6 WGI SPM (PDF): https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf
# - Park & Goodman (2023), Heat and Learning, PLOS Climate: https://journals.plos.org/climate/article?id=10.1371/journal.pclm.0000618
# - Goodman et al. (2018), Heat and Learning, NBER Working Paper 24639: https://www.nber.org/system/files/working_papers/w24639/w24639.pdf
# ==========================================

from __future__ import annotations

import io
import os
import time
import base64
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dup
from typing import Optional, Tuple, List

import streamlit as st
import plotly.express as px
from streamlit_echarts import st_echarts

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile

try:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    KAGGLE_AVAILABLE = True
except Exception:
    KAGGLE_AVAILABLE = False

# statsmodels(옵션) 사용 가능 여부: seaborn regplot(lowess=True)에 필요
try:
    import statsmodels.api as sm  # noqa: F401
    LOWESS_AVAILABLE = True
except Exception:
    LOWESS_AVAILABLE = False

# ---------- 페이지 설정 ----------
st.set_page_config(
    page_title="기온 × 학업 성취 대시보드 (공개데이터 + 사용자데이터)",
    layout="wide",
)

# ---------- 공통 상수 ----------
KST = ZoneInfo("Asia/Seoul")
TODAY_LOCAL = datetime.now(KST).date()  # “오늘(로컬 자정)” 이후 데이터 제거에 사용

REFERENCE_LINKS = [
    ("기후위키(기상청)", "http://www.climate.go.kr/home/10_wiki/index.php/%EA%B8%B0%ED%9B%84%EB%B3%80%ED%99%94%EC%9D%98_%EC%9B%90%EC%9D%B8"),
    ("한겨레(수면 연구 기사)", "https://www.hani.co.kr/arti/science/science_general/1046883.html"),
    ("IPCC AR6 WGI SPM", "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf"),
    ("PLOS Climate: Heat and Learning (2023)", "https://journals.plos.org/climate/article?id=10.1371/journal.pclm.0000618"),
    ("NBER: Heat and Learning (2018)", "https://www.nber.org/system/files/working_papers/w24639/w24639.pdf"),
]

# ---------- 폰트 적용 ----------
def try_apply_pretendard():
    """
    /fonts/Pretendard-Bold.ttf 가 있으면 Streamlit/Matplotlib/Plotly/ECharts에 적용.
    (추가 편의) /mnt/data/Pretendard_Bold.ttf 경로도 시도.
    없으면 자동 생략.
    """
    font_candidates = [
        "/fonts/Pretendard-Bold.ttf",
        "/mnt/data/Pretendard_Bold.ttf",  # 대화에서 제공된 경로 대응
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)
    if not font_path:
        return None

    # 웹(CSS) 임베드
    try:
        with open(font_path, "rb") as f:
            font_b64 = base64.b64encode(f.read()).decode("utf-8")
        css = f"""
        <style>
        @font-face {{
            font-family: 'Pretendard';
            src: url(data:font/ttf;base64,{font_b64}) format('truetype');
            font-weight: 700;
            font-style: normal;
        }}
        html, body, [class*="css"] {{
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', 'Helvetica Neue', Arial, sans-serif;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass

    # Matplotlib/Seaborn
    try:
        from matplotlib import font_manager
        font_manager.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = "Pretendard"
    except Exception:
        pass

    return "Pretendard"

FONT_FAMILY = try_apply_pretendard()

# ---------- 유틸 ----------
@st.cache_data(ttl=60 * 60)
def robust_get(url: str, params: dict | None = None, retry: int = 3, timeout: int = 20):
    """간단 재시도 GET. 실패 시 raise."""
    last_err = None
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
    """오늘(로컬 자정) 이후 자료 제거"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df[df[date_col] <= TODAY_LOCAL]

def to_csv_download(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("전처리 표 CSV 다운로드", csv, file_name=filename, mime="text/csv")

# ---------- 공개 데이터: NASA GISTEMP ----------
@st.cache_data(ttl=60 * 60)
def load_nasa_gistemp_monthly() -> pd.DataFrame:
    """
    NASA GISTEMP v4 전지구 월별 기온 이상(°C) 로드 → long 포맷(date, value, group).
    - 페이지: https://data.giss.nasa.gov/gistemp/
    - CSV:    https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    # 실패 대비 소형 예시(연구용 부적합)
    sample = """Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D
2024,1.30,1.25,1.33,1.30,1.29,1.30,1.31,1.28,1.22,1.20,1.25,1.30,1.29
2025,1.22,1.19,1.27,1.24,1.21,1.18,,,,,,,,
"""
    try:
        resp = robust_get(url)
        raw = resp.text
    except Exception:
        st.warning("NASA GISTEMP API 호출 실패 → 예시 데이터로 대체합니다. (연구/정책 판단에 사용 금지)")
        raw = sample

    # 앞/뒤 설명행 제거 + 헤더 정렬
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    header_idx = 0
    for i, ln in enumerate(lines):
        if ln.lower().startswith("year"):
            header_idx = i
            break
    clean = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(clean))

    # 요약열 제거
    for col in ["J-D", "D-N", "DJF", "MAM", "JJA", "SON"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # long 포맷
    df_long = df.melt(id_vars=["Year"], var_name="month", value_name="value")
    month_map = {m[:3].capitalize(): i for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
    df_long["month"] = df_long["month"].str[:3].str.capitalize().map(month_map)
    # NASA 원본에는 결측을 *** 등으로 표기하는 경우가 있어 숫자 변환 필요
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.dropna(subset=["month", "value"])
    df_long["date"] = pd.to_datetime(dict(year=df_long["Year"], month=df_long["month"], day=1))
    out = df_long[["date", "value"]].sort_values("date").reset_index(drop=True)
    out["group"] = "Global anomaly (°C)"
    out = drop_future(out, "date")
    return out

# ---------- 공개 데이터: World Bank EdStats (PISA) ----------
@st.cache_data(ttl=60 * 60)
def load_worldbank_indicator(countries: list[str], indicator: str) -> pd.DataFrame:
    """
    World Bank Indicators API에서 지표 로드 → long 포맷(date, value, group).
    - 문서: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
    - 참고: EdStats(교육지표) 데이터뱅크
    """
    base = "https://api.worldbank.org/v2/country/{}/indicator/{}"
    url = base.format(";".join([c.lower() for c in countries]), indicator)
    params = dict(format="json", per_page=20000)

    # 실패 대비 예시 (연구용 부적합)
    sample_csv = """country,value,date
KOR,524,2022
KOR,526,2018
KOR,554,2012
JPN,536,2022
JPN,527,2018
USA,465,2022
USA,478,2018
"""
    try:
        resp = robust_get(url, params=params)
        js = resp.json()
        if not isinstance(js, list) or len(js) < 2 or js[1] is None:
            raise RuntimeError("Unexpected response")
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
            raise RuntimeError("Empty data")
    except Exception:
        st.warning("World Bank API 호출 실패 → PISA 예시 데이터로 대체합니다. (연구/정책 판단에 사용 금지)")
        df = pd.read_csv(io.StringIO(sample_csv))
        df["date"] = pd.to_datetime(df["date"].astype(int).astype(str) + "-01-01")

    # 값 열을 확실히 숫자화하여 집계/Arrow 직렬화 오류 방지
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    df["group"] = df["country"]
    df = df[["date", "value", "group"]].sort_values(["group", "date"]).reset_index(drop=True)
    df = drop_future(df, "date")
    return df

# ---------- 사용자 데이터 표준화 ----------
def standardize_user_df(df: pd.DataFrame, date_col: str, value_col: str, group_col: str | None) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out["date"] = out[date_col].dt.date
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["value"])
    if group_col:
        out["group"] = out[group_col].astype(str)
    else:
        out["group"] = "default"
    out = out[["date", "value", "group"]].drop_duplicates()
    out = drop_future(out, "date")
    return out

# ---------- 레이아웃 ----------
st.title("📊 기온 상승 × 학업 성취 대시보드")
st.caption("공식 공개 데이터(NASA / World Bank) + 사용자 입력 데이터를 각각 시각화합니다. (로컬 기준 오늘 이후 데이터 자동 제거)")

# ----- 보고서 섹션: 서론 / 본론 / 결론 -----
st.markdown("## 서론 (문제 제기)")
st.markdown(
    """
기후 변화가 계속 됨에 따라 학생들이 기온 변화로 인해 평소 보다 더 큰 스트레스를 받는 것을 느꼈다. 점점 높아지는 최고기온과 길어지는 더위가 학생들의 학업 성적에 전혀 무관할까? 하는 궁금증을 가진 우리는 학생들의 학업에 미치는 영향에 관한 궁금증을 해소하고, 이를 학생들에게 알려 기온 상승 상황에서 성적 상승을 돕기위해서 이 주제를 선정해 연구한다.

인류 활동으로 발생한 지구 온실가스 배출량은 점점 증가하고 대기 중 이산화탄소 농도가  높아지며 전지구 평균기온도 높아지고 있다. 화석 연료 사용 증가 및 산업 발전과 인구 증가로 인해 과도한 에너지 소비가 일어나는 것이 주된 원인이다.
"""
)

# 서론 바로 아래: NASA 최근 24개월 라인 차트
try:
    _df_temp_intro = load_nasa_gistemp_monthly()
    _df_recent = _df_temp_intro.tail(24)
    _fig_intro = px.line(_df_recent, x="date", y="value", color="group", markers=True,
                         title="최근 24개월 전지구 기온 이상(°C)")
    if FONT_FAMILY:
        _fig_intro.update_layout(font_family="Pretendard")
    st.plotly_chart(_fig_intro, use_container_width=True)
except Exception as _e:
    st.warning(f"서론 시각화 로드 실패: {_e}")

st.markdown("## 본론 1 (데이터 분석)")
st.markdown(
    """
이번 연구에서는 기후 변화와 수면 시간의 관계를 분석하였다. 최근 기온 상승은 단순한 생활 불편이 아닌 청소년들의 수면 패턴에도 큰 영향을 주고 있다. 선 그래프를 통해 분석 한 결과, 기온이 높아질수록 평균 수면시간이 점차 줄어드는 경향이 확인되었다.
특히 더운 날씨에는 학생들이 깊은 잠에 드는 시간이 짧아지고, 자주 깨는 경우가 많아 수면의 질 또한 떨어지는 모습을 보였다. 이는 곧 수면 부족으로 이어지며, 학습 효율 저하와 집중력 감소의 원인으로 작용할 수 있다.
따라서 기후 변화는 단순히 환경적 위기만이 아니라, 청소년들의 수면과 학습 능력에 영향을 주는 중요한 요인임을 알 수 있다.
"""
)

# 본론1 바로 아래: 숫자만 있는 데이터 → 즉시 그래프
st.markdown("#### 숫자 리스트 즉시 시각화")
_num_text = st.text_area(
    "숫자 리스트를 콤마 또는 줄바꿈으로 입력 (예: 3, 5, 2, 7)",
    height=100,
    placeholder="3, 5, 2, 7, 6, 4"
)
if _num_text.strip():
    try:
        _tokens = [t.strip() for t in _num_text.replace("\n", ",").split(",") if t.strip()]
        _values = [float(t) for t in _tokens]
        _df_nums = pd.DataFrame({"idx": range(1, len(_values)+1), "value": _values})
        _fig_line = px.line(_df_nums, x="idx", y="value", markers=True, title="입력 숫자 라인 차트")
        if FONT_FAMILY:
            _fig_line.update_layout(font_family="Pretendard")
        st.plotly_chart(_fig_line, use_container_width=True)

        _fig_hist = px.histogram(_df_nums, x="value", nbins=min(20, max(5, len(_values)//2)), title="분포(히스토그램)")
        if FONT_FAMILY:
            _fig_hist.update_layout(font_family="Pretendard")
        st.plotly_chart(_fig_hist, use_container_width=True)
    except Exception:
        st.info("숫자만 입력해 주세요. 예: 1, 2, 3, 4")

# 본론1 바로 아래: Kaggle.json 자동 인증 + 간편 다운로드/그래프
if KAGGLE_AVAILABLE:
    st.markdown("#### Kaggle 데이터 빠른 시각화")
    _auto_api = None
    try:
        _auto_api = KaggleApi()
        # 자동 경로: 루트의 Kaggle.json/kaggle.json 또는 ~/.kaggle/kaggle.json
        _candidates = [
            os.path.join(os.getcwd(), "Kaggle.json"),
            os.path.join(os.getcwd(), "kaggle.json"),
        ]
        _root_kg = next((p for p in _candidates if os.path.exists(p)), None)
        if _root_kg:
            tmp_dir = "/tmp/kaggle_creds_auto"
            os.makedirs(tmp_dir, exist_ok=True)
            cred_path = os.path.join(tmp_dir, "kaggle.json")
            with open(_root_kg, "rb") as src, open(cred_path, "wb") as dst:
                dst.write(src.read())
            os.chmod(cred_path, 0o600)
            os.environ["KAGGLE_CONFIG_DIR"] = tmp_dir
            _auto_api.authenticate()
        else:
            # 홈 디렉터리 기본 사용
            _auto_api.authenticate()
    except Exception as _e:
        st.info(f"Kaggle 자동 인증 생략: {_e}")
        _auto_api = None

    _col_k1, _col_k2 = st.columns([2, 1])
    with _col_k1:
        _ds_quick = st.text_input("Dataset slug (owner/dataset)", placeholder="zynicide/wine-reviews", key="quick_ds")
        _out_quick = st.text_input("다운로드 폴더", value="/tmp/kaggle_quick", key="quick_out")
        _btn_quick = st.button("Kaggle에서 받아서 그리기", disabled=not _auto_api or not _ds_quick)
        if _btn_quick and _auto_api:
            try:
                os.makedirs(_out_quick, exist_ok=True)
                _auto_api.dataset_download_files(_ds_quick, path=_out_quick, quiet=True, force=True)
                zips = [f for f in os.listdir(_out_quick) if f.endswith('.zip')]
                for z in zips:
                    with ZipFile(os.path.join(_out_quick, z), 'r') as zf:
                        zf.extractall(_out_quick)
                csvs = [os.path.join(_out_quick, f) for f in os.listdir(_out_quick) if f.lower().endswith('.csv')]
                if not csvs:
                    st.warning("CSV 파일을 찾지 못했습니다.")
                else:
                    st.session_state["_kg_quick_csvs"] = csvs
                    st.success(f"다운로드 완료 (CSV {len(csvs)}개)")
            except Exception as _e:
                st.error(f"Kaggle 다운로드 실패: {_e}")
    with _col_k2:
        _csv_list = st.session_state.get("_kg_quick_csvs", [])
        if _csv_list:
            _pick = st.selectbox("CSV 선택", _csv_list, key="quick_csv_pick")
            try:
                _dfk = pd.read_csv(_pick)
                st.dataframe(_dfk.head(20), width='stretch')
                cols = list(_dfk.columns)
                c1, c2, c3 = st.columns(3)
                date_col = c1.selectbox("날짜 열", options=cols, key="quick_date")
                value_col = c2.selectbox("값 열", options=cols, key="quick_value")
                group_col_opt = c3.selectbox("그룹 열(선택)", options=["<없음>"] + cols, key="quick_group")
                group_col = None if group_col_opt == "<없음>" else group_col_opt
                std = standardize_user_df(_dfk, date_col, value_col, group_col)
                st.success("전처리 완료")
                fig_quick = px.line(std, x="date", y="value", color="group", markers=True, title="Kaggle 데이터 라인 차트")
                if FONT_FAMILY:
                    fig_quick.update_layout(font_family="Pretendard")
                st.plotly_chart(fig_quick, use_container_width=True)
            except Exception as _e:
                st.error(f"CSV 파싱 실패: {_e}")

st.markdown("## 본론 2 (원인 및 영향 탐구)")
st.markdown(
    """
기온 변화와 성적은 실제로 상관관계를 가진다. 아래의 막대 그래프와 산점도를 살피면 더 정확히 알 수 있다.

위의 막대그래프는 전 세계 다양한 연구에서 보고된 기온 상승과 학업 성적 변화의 상관관계를 비교한 것이다. 그래프에서 볼 수 있듯이, 대부분의 연구에서 기온이 일정 수준 이상 상승하면 학생들의 성적이 표준편차 단위로 감소하는 경향이 나타난다.
특히 OECD 국제학업성취도 평가(PISA)를 활용한 58개국 분석에서는 고온 노출이 누적될수록 성적이 크게 하락하는 결과가 확인되었다. 미국과 뉴욕의 사례 또한 시험 당일 기온이 높을수록 학생들의 성적과 합격률이 유의하게 떨어졌다. 한국의 경우 단일 고온일의 효과는 비교적 작지만, 34℃ 이상의 날이 누적될수록 수학과 영어 성적이 점차 감소하는 경향을 보였다.
이처럼 고온 환경은 단순한 불쾌감을 넘어 학업 성취에도 부정적인 영향을 미치며, 특히 누적 효과가 장기적인 성적 저하로 이어질 수 있음을 시사한다.

위 그림은 2012년 PISA(패널 A) 또는 SEDA(패널 B) 수학 평균 점수와 국가 또는 미국 카운티별 연평균 기온의 산점도이다.

연평균 기온은 1980년부터 2011년까지 측정되었고, 패널 B는 평균 기온 분포의 백분위 별로 표준화된 3~8학년 수학 점수(2009~2013년)의 구간별 백분위 그래프이다. 점수는 과목, 학년, 연도별로 표준화된 점수를 사용한다.

이 산점도는 간단히 말하자면 미국 학생들의 수학 성적이 기온에 따라 어떻게 변화하는지 보여준다. 위 그래프는 기온이 높아질 수록 성적이 하락하는 경향을 면밀히 보여주고 있다. 따라서 우리는 기온 상승이 학생들의 성적에 밀접한 연관을 가진다는 사실을 알 수 있다.
"""
)

# 본론2 바로 아래: PISA × 기온 상관 산점도(요약)
try:
    _countries_default = ["KOR", "JPN", "USA"]
    _indicator_default = "LO.PISA.MAT"
    _country_pick = st.selectbox("국가 선택 (요약 산점도)", _countries_default, index=0, key="intro_country")
    _df_pisa_intro = load_worldbank_indicator(_countries_default, _indicator_default)
    if not _df_pisa_intro.empty:
        _annual_temp = load_nasa_gistemp_monthly().assign(year=lambda d: pd.to_datetime(d["date"]).dt.year) \
                                               .groupby("year", as_index=False)["value"].mean() \
                                               .rename(columns={"value": "temp_anom"})
        _pisa_focus = _df_pisa_intro[_df_pisa_intro["group"] == _country_pick].copy()
        _pisa_focus["year"] = pd.to_datetime(_pisa_focus["date"]).dt.year
        _merged_intro = pd.merge(_pisa_focus[["year", "value"]], _annual_temp, on="year", how="inner")
        _merged_intro = _merged_intro.rename(columns={"value": "pisa_score"})
        _series_intro = [[float(r["temp_anom"]), float(r["pisa_score"])] for _, r in _merged_intro.iterrows()]
        _opt_intro = {
            "title": {"text": f"{_country_pick}: 기온 이상 vs PISA(요약)", "left": "center",
                      "textStyle": {"fontFamily": FONT_FAMILY or "inherit"}},
            "tooltip": {"trigger": "item"},
            "xAxis": {"name": "연평균 기온 이상(°C)", "nameLocation": "middle", "nameGap": 28},
            "yAxis": {"name": "PISA 평균 점수", "nameLocation": "middle", "nameGap": 28},
            "series": [{"type": "scatter", "data": _series_intro, "symbolSize": 10}],
        }
        st_echarts(options=_opt_intro, height="360px")
    else:
        st.info("요약 산점도를 표시할 PISA 데이터가 없습니다.")
except Exception as _e:
    st.warning(f"본론2 요약 산점도 생성 실패: {_e}")

tab_pub, tab_user = st.tabs(["🌍 공식 공개 데이터 대시보드", "🧑‍💻 사용자 입력 대시보드"])
tab_kaggle = None
if KAGGLE_AVAILABLE:
    tab_pub, tab_user, tab_kaggle = st.tabs(["🌍 공식 공개 데이터 대시보드", "🧑‍💻 사용자 입력 대시보드", "📦 Kaggle 데이터 불러오기"])

# ==========================
# 1) 공식 공개 데이터
# ==========================
with tab_pub:
    st.subheader("① 전지구 월별 기온 이상 (NASA GISTEMP v4)")
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        df_temp = load_nasa_gistemp_monthly()
        st.write("데이터 표 (전처리·표준화 완료)")
        st.dataframe(df_temp.tail(24), width='stretch')

        fig = px.line(
            df_temp, x="date", y="value", color="group",
            title="전지구 기온 이상(월별, °C)"
        )
        if FONT_FAMILY:
            fig.update_layout(font_family="Pretendard")
        st.plotly_chart(fig, use_container_width=True)
        to_csv_download(df_temp, "nasa_gistemp_monthly.csv")

    with colB:
        annual = df_temp.copy()
        annual["value"] = pd.to_numeric(annual["value"], errors="coerce")
        annual = annual.dropna(subset=["value"])  # 안전 처리
        annual["year"] = pd.to_datetime(annual["date"]).dt.year
        annual = annual.groupby(["year", "group"], as_index=False)["value"].mean()
        fig2 = px.bar(annual.tail(30), x="year", y="value", color="group", title="최근 30년 연평균 이상(°C)")
        if FONT_FAMILY:
            fig2.update_layout(font_family="Pretendard")
        st.plotly_chart(fig2, use_container_width=True)

    fig3, ax = plt.subplots(figsize=(5.6, 3.2))
    # statsmodels 없거나 오류 시 폴백: 일반 회귀선으로 표시
    try:
        sns.regplot(data=annual, x="year", y="value", lowess=LOWESS_AVAILABLE, ax=ax)
    except Exception as e:
        sns.regplot(data=annual, x="year", y="value", lowess=False, ax=ax)
        st.warning("LOWESS 선을 그릴 수 없어 일반 회귀선으로 대체했습니다. (statsmodels 필요)")
    ax.set_title("연평균 이상 추세(회귀/LOWESS)")
    st.pyplot(fig3, clear_figure=True)

    st.markdown("> 출처: NASA GISTEMP v4 (Tables of Global and Hemispheric Monthly Means, Global-mean monthly CSV)")

    st.markdown("---")
    st.subheader("② PISA 평균 성취 (World Bank EdStats, Indicators API)")
    countries_default = ["KOR", "JPN", "USA"]
    c_sel = st.multiselect("국가(ISO3, 여러 개 선택 가능)",
                           options=countries_default + ["DEU", "CHN", "GBR", "FRA", "CAN", "AUS", "NZL", "SGP", "FIN"],
                           default=countries_default)
    indicator = st.selectbox(
        "지표 선택 (EdStats)",
        options=["LO.PISA.MAT", "LO.PISA.REA", "LO.PISA.SCI"],
        index=0,
        help="World Bank Indicators API의 EdStats 지표 코드(국가/년도 가용성은 차이가 있을 수 있음).",
    )
    df_pisa = load_worldbank_indicator(c_sel or countries_default, indicator)

    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        fig4 = px.line(df_pisa, x="date", y="value", color="group", markers=True,
                       title=f"{indicator} 연도별 추이 (PISA 평균 점수)")
        if FONT_FAMILY:
            fig4.update_layout(font_family="Pretendard")
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(df_pisa.head(20), width='stretch')
        to_csv_download(df_pisa, f"worldbank_{indicator.lower()}.csv")

    with c2:
        if not df_pisa.empty:
            annual_temp = load_nasa_gistemp_monthly().assign(year=lambda d: pd.to_datetime(d["date"]).dt.year) \
                                                     .groupby("year", as_index=False)["value"].mean() \
                                                     .rename(columns={"value": "temp_anom"})
            focus = (c_sel or countries_default)[0]
            pisa_focus = df_pisa[df_pisa["group"] == focus].copy()
            pisa_focus["year"] = pd.to_datetime(pisa_focus["date"]).dt.year
            merged = pd.merge(pisa_focus[["year", "value"]], annual_temp, on="year", how="inner")
            merged = merged.rename(columns={"value": "pisa_score"})
            st.write(f"상관 보기: {focus} (PISA) × 전지구 연평균 기온 이상")
            series_data = [[float(r["temp_anom"]), float(r["pisa_score"])] for _, r in merged.iterrows()]
            option = {
                "title": {"text": f"{focus}: 기온 이상 vs PISA", "left": "center",
                          "textStyle": {"fontFamily": FONT_FAMILY or "inherit"}},
                "tooltip": {"trigger": "item"},
                "xAxis": {"name": "연평균 기온 이상(°C)", "nameLocation": "middle", "nameGap": 28},
                "yAxis": {"name": "PISA 평균 점수", "nameLocation": "middle", "nameGap": 28},
                "series": [{"type": "scatter", "data": series_data, "symbolSize": 10}],
            }
            st_echarts(options=option, height="350px")
            st.dataframe(merged.sort_values("year", ascending=False), width='stretch')
        else:
            st.info("표시할 PISA 데이터가 없습니다.")

    with st.expander("📚 참고 자료 링크"):
        st.markdown("\n".join([f"- [{name}]({url})" for name, url in REFERENCE_LINKS]))

# ==========================
# 2) 사용자 입력 대시보드
# ==========================
with tab_user:
    st.subheader("CSV 업로드/붙여넣기 → 표준화 → 시각화")

    up_col, paste_col = st.columns(2)
    with up_col:
        file = st.file_uploader("CSV 업로드 (UTF-8 권장)", type=["csv"])
        df_user = None
        if file:
            try:
                df_user = pd.read_csv(file)
            except Exception:
                st.error("CSV 읽기 실패: 인코딩/구분자 확인")
    with paste_col:
        txt = st.text_area("CSV 내용 붙여넣기 (옵션)", height=160,
                           placeholder="date,value,group\n2024-06-01,7.3,A\n2024-06-02,6.9,A\n...")
        if txt.strip() and df_user is None:
            try:
                df_user = pd.read_csv(io.StringIO(txt))
            except Exception:
                st.error("붙여넣기 CSV 파싱 실패")

    if df_user is not None and not df_user.empty:
        st.write("원본 미리보기")
        st.dataframe(df_user.head(20), width='stretch')

        cols = list(df_user.columns)
        c1, c2, c3 = st.columns(3)
        date_col = c1.selectbox("날짜 열 선택", options=cols)
        value_col = c2.selectbox("값 열 선택", options=cols)
        group_col_opt = c3.selectbox("그룹 열 선택(선택)", options=["<없음>"] + cols)
        group_col = None if group_col_opt == "<없음>" else group_col_opt

        std = standardize_user_df(df_user, date_col, value_col, group_col)
        st.success("전처리 완료 (결측/형변환/중복 제거 + 미래데이터 제거)")
        st.dataframe(std.head(30), width='stretch')
        to_csv_download(std, "user_standardized.csv")

        st.markdown("#### 시각화")
        t1, t2 = st.columns([2, 1], gap="large")
        with t1:
            figu = px.line(std, x="date", y="value", color="group", markers=True, title="라인 차트")
            if FONT_FAMILY:
                figu.update_layout(font_family="Pretendard")
            st.plotly_chart(figu, use_container_width=True)
        with t2:
            agg_unit = st.selectbox("집계 단위", ["일", "월(평균)", "연(평균)"], index=1)
            tmp = std.copy()
            if agg_unit == "월(평균)":
                tmp["date"] = pd.to_datetime(tmp["date"]).dt.to_period("M").dt.to_timestamp()
            elif agg_unit == "연(평균)":
                tmp["date"] = pd.to_datetime(tmp["date"]).dt.to_period("Y").dt.to_timestamp()
            agg = tmp.groupby(["date", "group"], as_index=False)["value"].mean()
            figb = px.bar(agg, x="date", y="value", color="group", title=f"집계 막대 ({agg_unit})")
            if FONT_FAMILY:
                figb.update_layout(font_family="Pretendard")
            st.plotly_chart(figb, use_container_width=True)

        st.markdown("#### 산점도(그룹별) + 간단 회귀선")
        g_list = sorted(std["group"].unique())
        g_pick = st.selectbox("그룹 선택", g_list)
        sub = std[std["group"] == g_pick].copy()
        sub["x"] = pd.to_datetime(sub["date"]).map(pd.Timestamp.toordinal)
        if len(sub) >= 2:
            m, b = np.polyfit(sub["x"], sub["value"], 1)
            sub = sub.sort_values("x")
            sub["trend"] = m * sub["x"] + b
            series_scatter = [[int(x), float(y)] for x, y in zip(sub["x"], sub["value"])]
            series_line = [[int(x), float(y)] for x, y in zip(sub["x"], sub["trend"])]
            option2 = {
                "title": {"text": f"{g_pick} 산점/회귀", "left": "center",
                          "textStyle": {"fontFamily": FONT_FAMILY or "inherit"}},
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "value", "axisLabel": {"formatter": "{value} (ordinal date)"}},
                "yAxis": {"type": "value", "name": "value"},
                "series": [
                    {"type": "scatter", "name": "data", "data": series_scatter, "symbolSize": 9},
                    {"type": "line", "name": "trend", "data": series_line, "smooth": True},
                ],
                "legend": {"top": 0},
            }
            st_echarts(options=option2, height="360px")
        else:
            st.info("산점도/회귀를 위해서는 최소 2개 이상의 데이터가 필요합니다.")
    else:
        st.info("CSV 업로드 또는 붙여넣기로 사용자 데이터를 제공하세요.")

# ==========================
# 3) Kaggle 데이터 불러오기 (옵션)
# ==========================
if KAGGLE_AVAILABLE and tab_kaggle is not None:
    with tab_kaggle:
        st.subheader("Kaggle 데이터셋 다운로드 → CSV 선택 → 표준화")
        st.caption("자격 증명은 세션 내 메모리 또는 ~/.kaggle/kaggle.json을 사용합니다.")

        col_auth, col_dl = st.columns([1, 2])
        with col_auth:
            st.markdown("#### 1) 인증")
            use_file = st.toggle("kaggle.json 업로드", value=False, help="체크 시 아래 업로더 사용. 미체크 시 사용자/키 입력")
            kaggle_json = None
            username: Optional[str] = None
            key: Optional[str] = None
            if use_file:
                up = st.file_uploader("kaggle.json 업로드", type=["json"])
                if up is not None:
                    kaggle_json = up.read()
            else:
                username = st.text_input("Kaggle Username", value=os.environ.get("KAGGLE_USERNAME", ""))
                key = st.text_input("Kaggle Key", type="password", value=os.environ.get("KAGGLE_KEY", ""))

            def do_auth() -> Tuple[Optional[KaggleApi], Optional[str]]:
                try:
                    api = KaggleApi()
                    if kaggle_json:
                        # 세션 임시 파일에 저장하여 인증
                        tmp_dir = st.session_state.get("_kg_tmp", None)
                        if not tmp_dir:
                            tmp_dir = "/tmp/kaggle_creds"
                            os.makedirs(tmp_dir, exist_ok=True)
                            st.session_state["_kg_tmp"] = tmp_dir
                        cred_path = os.path.join(tmp_dir, "kaggle.json")
                        with open(cred_path, "wb") as f:
                            f.write(kaggle_json)
                        os.chmod(cred_path, 0o600)
                        os.environ["KAGGLE_CONFIG_DIR"] = tmp_dir
                        api.authenticate()
                        return api, cred_path
                    elif username and key:
                        os.environ["KAGGLE_USERNAME"] = username
                        os.environ["KAGGLE_KEY"] = key
                        api.authenticate()
                        return api, None
                    else:
                        return None, None
                except Exception as e:
                    st.error(f"인증 실패: {e}")
                    return None, None

            api, cred_file = do_auth()
            if api:
                st.success("Kaggle 인증 성공")
            else:
                st.info("인증이 필요합니다. 좌측에서 kaggle.json 업로드 또는 계정/키 입력 후 재시도하세요.")

        with col_dl:
            st.markdown("#### 2) 데이터셋 다운로드")
            ds = st.text_input("Dataset slug (owner/dataset)", placeholder="zynicide/wine-reviews")
            out_dir = st.text_input("다운로드 폴더", value="/tmp/kaggle_downloads")
            dl_btn = st.button("다운로드", disabled=not api or not ds)
            csv_files: List[str] = []
            selected_csv: Optional[str] = None
            if dl_btn and api:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    api.dataset_download_files(ds, path=out_dir, quiet=False, force=True)
                    # zip 파일 찾기 후 해제
                    zips = [f for f in os.listdir(out_dir) if f.endswith('.zip')]
                    for z in zips:
                        with ZipFile(os.path.join(out_dir, z), 'r') as zf:
                            zf.extractall(out_dir)
                    csv_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith('.csv')]
                    st.session_state["_kaggle_csvs"] = csv_files
                    st.success(f"다운로드 완료. CSV {len(csv_files)}개 발견")
                except Exception as e:
                    st.error(f"다운로드/해제 실패: {e}")

            csv_files = st.session_state.get("_kaggle_csvs", [])
            if csv_files:
                selected_csv = st.selectbox("CSV 선택", csv_files)
                if selected_csv and os.path.exists(selected_csv):
                    try:
                        df_k = pd.read_csv(selected_csv)
                        st.write("미리보기")
                        st.dataframe(df_k.head(30), width='stretch')
                        # 열 매핑 → 기존 표준화 함수 재사용
                        cols = list(df_k.columns)
                        c1, c2, c3 = st.columns(3)
                        date_col = c1.selectbox("날짜 열 선택", options=cols, key="kg_date")
                        value_col = c2.selectbox("값 열 선택", options=cols, key="kg_value")
                        group_col_opt = c3.selectbox("그룹 열 선택(선택)", options=["<없음>"] + cols, key="kg_group_opt")
                        group_col = None if group_col_opt == "<없음>" else group_col_opt

                        std = standardize_user_df(df_k, date_col, value_col, group_col)
                        st.success("표준화 완료")
                        st.dataframe(std.head(30), width='stretch')
                        to_csv_download(std, os.path.basename(selected_csv).replace('.csv', '_standardized.csv'))

                        # 재사용 시각화
                        st.markdown("#### 시각화")
                        figu = px.line(std, x="date", y="value", color="group", markers=True, title="라인 차트")
                        if FONT_FAMILY:
                            figu.update_layout(font_family="Pretendard")
                        st.plotly_chart(figu, use_container_width=True)

                        tmp = std.copy()
                        tmp["date"] = pd.to_datetime(tmp["date"]).dt.to_period("M").dt.to_timestamp()
                        agg = tmp.groupby(["date", "group"], as_index=False)["value"].mean()
                        figb = px.bar(agg, x="date", y="value", color="group", title="월 평균 막대")
                        if FONT_FAMILY:
                            figb.update_layout(font_family="Pretendard")
                        st.plotly_chart(figb, use_container_width=True)

                    except Exception as e:
                        st.error(f"CSV 파싱 실패: {e}")
            else:
                st.info("CSV가 보이지 않으면 데이터셋 슬러그를 확인하고 다시 다운로드하세요.")

# ---------- 결론 (제언) ----------
st.markdown("## 결론 (제언)")
st.markdown(
    """
이번 연구를 통해 우리는 기온 상승이 단순히 생활 불편에 그치지 않고, 학생들의 수면 질 저하와 집중력 감소를 초래하며, 학업 성취도에 부정적인 영향을 줄 수 있음을 확인하였다. 특히 기온이 일정 수준 이상 상승할 경우 성적이 표준편차 단위로 감소하는 경향이 여러 연구에서 공통적으로 드러났다. 이는 단일 요인이 아닌, 반복적이고 누적된 높은 온도 노출이 장기적으로 학생들의 학습 능력을 저해한다는 점을 보여준다.
"""
)

# ---------- 하단 정보 ----------
st.markdown("---")
st.caption(
    "본 앱은 공개 데이터 실패 시 예시 데이터로 자동 대체하며, 예시 데이터는 연구/정책 판단에 부적합합니다. "
    "PISA 데이터는 World Bank EdStats API 가용성에 따라 국가·연도별 공백이 있을 수 있습니다."
)
