# Road Safety BI Report (Streamlit)

Interactive Power BI style dashboard built for the ST2DLDI Data Integration & Applications case study, based on the French Road Safety Open Data (data.gouv.fr, 2024 edition).

## What it contains

- **Overview**: KPI cards (accidents, users involved, killed, vehicles, fatality rate), accidents by month, severity breakdown, and an interactive accident map.
- **Data Quality**: the quality scorecard (completeness, uniqueness, validity), the missing / non-response rate chart, the GPS validity check, and the duplicate row count, all recomputed live from the raw files.
- **Accident Analysis**: accidents by time of day and weather condition, fatality rate by road category, top departments, and collision type distribution.
- **User Profile**: KPIs by user category (driver, passenger, pedestrian), severity by age group, and severity by sex.
- **Table Explorer**: a per-table view of the four raw Bronze files (`caract`, `lieux`, `usagers`, `vehicules`) independently of the merged fact table used elsewhere: row/column counts, missing cells, duplicate rows, a full column summary (dtype, missing %, distinct values), and an interactive distribution chart for any column the user picks.

The Overview, Data Quality, Accident Analysis and User Profile pages share the same sidebar filters (zone, time of day, severity, weather), so every KPI and chart updates together, the same way a Power BI report page would behave. The Table Explorer page works on the raw tables directly and is not affected by these filters.

## How to run it locally

1. Make sure the four source files are present in the `data/` folder next to `app.py`:
   `caract-2024.csv`, `lieux-2024.csv`, `usagers-2024.csv`, `vehicules-2024.csv`.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```
   streamlit run app.py
   ```
4. Streamlit opens the report at `http://localhost:8501` in your browser.

## Publishing it online (optional)

This sandbox cannot expose a permanent public URL, so the app is delivered as source code to run locally. To get a shareable link, the simplest option is Streamlit Community Cloud (share.streamlit.io): push this folder to a GitHub repository, connect it from the Streamlit Cloud dashboard, and point it at `app.py`. Once deployed, the resulting URL can be added to the notebook in place of the placeholder link.
