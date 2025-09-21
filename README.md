# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Run the Streamlit app (local)

1. Install the requirements

   ```
   pip install -r requirements.txt
   ```

2. Run the app

   ```
   streamlit run streamlit_app.py
   ```

   - If the default port `8501` is busy, try another port:

   ```
   streamlit run streamlit_app.py --server.port 8502
   ```

### Troubleshooting

- Arrow serialization error for dataframes (type mix like float + str):
  - Fixed in code by enforcing numeric conversion for value columns, but if you edit the app, ensure
    columns used for aggregation are numeric: `pd.to_numeric(df["value"], errors="coerce")` then drop `NaN`.
- Deprecation warning: `use_container_width` → Use `width='stretch'`.

### Kaggle integration (optional)

The app includes a Kaggle tab (visible when the `kaggle` package is installed) for downloading and previewing Kaggle datasets.

1) Credentials
   - EITHER upload `kaggle.json` in the Kaggle tab, OR set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
   - Docker Compose: copy `.env.example` to `.env` and fill `KAGGLE_USERNAME`/`KAGGLE_KEY`, or export them before `docker compose up`.
   - You may also place `Kaggle.json` at the repo root; the app copies it to `.kaggle/kaggle.json` automatically.
   - On Linux/macOS you can instead place at `~/.kaggle/kaggle.json` with permissions 600.

2) Use in the app
   - Open the Kaggle tab, enter a dataset slug like `owner/dataset` (e.g., `zynicide/wine-reviews`).
   - Click "파일 목록 조회" to see files; click "데이터셋 다운로드" to fetch and unzip into `kaggle_data/`.
   - Choose a CSV to preview; optionally map date/value/(group) to the standard schema and download a standardized CSV.

Security notes
- Never commit your Kaggle API key to source control. Keep `Kaggle.json` private.
- In containerized/cloud environments, prefer environment variables or uploading `kaggle.json` per session.

---

## New analytics in this template

- Yearly temperature rise (YoY): In the Public Data tab, the NASA GISTEMP monthly anomalies are aggregated by year and a bar chart shows year-over-year change (°C). Positive bars indicate warming vs previous year.
- Snow depth by temperature × year: In the User Data tab, map your date/temperature/snow-depth columns. Choose bin width for temperature (e.g., 2°C) and aggregation (average or sum). The app renders a heatmap of Year × Temperature-bin with snow depth as color, plus drill-down:
   - Pick a year to see snow by temperature-bin
   - Pick a temperature-bin to see snow trend over years

Tips
- Ensure temperature and snow-depth columns are numeric or convertible. Non-numeric entries are dropped.
- If you have groups (e.g., station, region), select a group column to compare groups one at a time in the snow view.

## Full Docker stack: Postgres + Metabase + Streamlit + Loader

This repo now includes a compose stack that loads real data into Postgres and exposes Metabase for an interactive dashboard UI.

What you get
   - nasa_gistemp_monthly(date, value, metric)
   - worldbank_indicators(country, indicator, date, value)
- Metabase at http://localhost:3000 to explore and build dashboards
- A one-shot loader container that fetches real data from NASA GISTEMP and World Bank EdStats APIs and upserts into Postgres

1) Build and start everything
docker compose up -d --build
2) Wait ~20–60s for db to be healthy and loader to finish. Check logs if needed:
docker logs blankapp_loader --tail=100
```
3) Open Metabase: http://localhost:3000
       - Host: db
       - Port: 5432
       - DB name: appdb
       - Username: appuser
       - Password: apppassword
    - After saving, browse the tables and view to create questions/dashboards.

Bring the stack down
```

Environment knobs (optional)
- Edit `docker-compose.yml` to change countries/indicators via `WB_COUNTRIES` and `WB_INDICATORS` on the loader service.

Why Metabase? Popular dockerized dashboard UIs (top 5)
- Metabase (we use this): super simple, great auto-modeling, OSS/free, first-class Docker image.
- Apache Superset: powerful SQL+viz, enterprise-grade; heavier to set up, but excellent for complex analytics.
- Grafana: time-series first; great with Postgres too, but best for infra metrics.
- Redash: lightweight SQL query + dashboard; community edition is popular.
- Budibase: app builder with data sources; flexible but broader scope than pure analytics.

For this project, Metabase offers the fastest path to a clean analytics UI over Postgres with minimal config.

Notes
- The loader uses only actual data from NASA and the World Bank (no synthetic fallback). If those APIs are temporarily unavailable, the loader will fail—re-run later.
- If you need Metabase to use Postgres for its own metadata (recommended for production), set MB_DB_* env vars and add a dependency on `db`.
