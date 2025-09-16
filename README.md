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

This app can download datasets from Kaggle directly if `kaggle` package and credentials are available.

1. Credentials
   - Upload `kaggle.json` in the Kaggle tab, or
   - Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
   - Alternatively place `kaggle.json` in `~/.kaggle/kaggle.json` (chmod 600).

2. Use the Kaggle tab
   - Enter dataset slug like `owner/dataset` (e.g., `zynicide/wine-reviews`).
   - Click Download. The app will unzip and list CSV files for selection.
   - Map date/value/(optional) group columns → standardize → visualize.

---

## Full Docker stack: Postgres + Metabase + Streamlit + Loader

This repo now includes a compose stack that loads real data into Postgres and exposes Metabase for an interactive dashboard UI.

What you get
- Postgres 16 (port 5432) with two tables and one view:
   - nasa_gistemp_monthly(date, value, metric)
   - worldbank_indicators(country, indicator, date, value)
   - vw_pisa_vs_temp (JOIN of PISA indicators and annualized NASA anomalies)
- Metabase at http://localhost:3000 to explore and build dashboards
- Your existing Streamlit app at http://localhost:8501
- A one-shot loader container that fetches real data from NASA GISTEMP and World Bank EdStats APIs and upserts into Postgres

Quick start (Windows cmd)
1) Build and start everything
```
docker compose up -d --build
```
2) Wait ~20–60s for db to be healthy and loader to finish. Check logs if needed:
```
docker logs blankapp_loader --tail=100
```
3) Open Metabase: http://localhost:3000
    - Initial setup (email/password of your choice)
    - Add a database connection:
       - Database: Postgres
       - Host: db
       - Port: 5432
       - DB name: appdb
       - Username: appuser
       - Password: apppassword
    - After saving, browse the tables and view to create questions/dashboards.

4) Open Streamlit: http://localhost:8501

Bring the stack down
```
docker compose down
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
