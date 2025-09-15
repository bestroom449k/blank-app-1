# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

   - If the default port `8501` is busy, try another port:

   ```
   $ streamlit run streamlit_app.py --server.port 8502
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
