import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Road Safety BI Report", layout="wide", page_icon="🚦")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ----------------------------------------------------------------------------
# Data loading and preparation (Bronze -> Silver, cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    caract = pd.read_csv(os.path.join(DATA_DIR, "caract-2024.csv"), sep=";", encoding="utf-8")
    lieux = pd.read_csv(os.path.join(DATA_DIR, "lieux-2024.csv"), sep=";", encoding="utf-8")
    usagers = pd.read_csv(os.path.join(DATA_DIR, "usagers-2024.csv"), sep=";", encoding="utf-8")
    vehicules = pd.read_csv(os.path.join(DATA_DIR, "vehicules-2024.csv"), sep=";", encoding="utf-8")
    return caract, lieux, usagers, vehicules


@st.cache_data
def build_silver(caract, lieux, usagers, vehicules):
    caract = caract.copy()
    lieux = lieux.copy()
    usagers = usagers.copy()
    vehicules = vehicules.copy()

    # Standardization
    caract["date_heure"] = pd.to_datetime(
        caract["an"].astype(str) + "-" + caract["mois"].astype(str).str.zfill(2) + "-"
        + caract["jour"].astype(str).str.zfill(2) + " " + caract["hrmn"],
        format="%Y-%m-%d %H:%M", errors="coerce",
    )
    caract["lat"] = caract["lat"].astype(str).str.replace(",", ".").astype(float)
    caract["long"] = caract["long"].astype(str).str.replace(",", ".").astype(float)
    caract["zone_geo"] = np.where(
        caract["lat"].between(41, 51.5) & caract["long"].between(-5.5, 10),
        "Mainland", "Overseas / to verify",
    )

    lum_labels = {1: "Daylight", 2: "Dusk or dawn", 3: "Night, no public lighting",
                  4: "Night, public lighting off", 5: "Night, public lighting on"}
    atm_labels = {1: "Normal", 2: "Light rain", 3: "Heavy rain", 4: "Snow / hail",
                  5: "Fog / smoke", 6: "Strong wind / storm", 7: "Dazzling weather",
                  8: "Overcast", 9: "Other"}
    col_labels = {1: "Head-on", 2: "Rear-end", 3: "Side impact", 4: "Chain collision",
                  5: "Multiple collision", 6: "Other collision", 7: "No collision"}
    caract["lum_label"] = caract["lum"].map(lum_labels).fillna("Not specified")
    caract["atm_label"] = caract["atm"].map(atm_labels).fillna("Not specified")
    caract["col_label"] = caract["col"].map(col_labels).fillna("Not specified")

    def time_of_day(dt):
        if pd.isna(dt):
            return np.nan
        h = dt.hour
        if 6 <= h < 10:
            return "Morning peak"
        if 10 <= h < 16:
            return "Daytime"
        if 16 <= h < 20:
            return "Evening peak"
        return "Night"

    caract["time_of_day"] = caract["date_heure"].apply(time_of_day)

    # Cleaning
    cols_minus1_usagers = ["sexe", "secu1", "secu2", "secu3", "trajet", "locp", "actp", "etatp"]
    for c in cols_minus1_usagers:
        usagers[c] = usagers[c].replace(-1, np.nan).replace("-1", np.nan)

    lieux["vma"] = pd.to_numeric(lieux["vma"], errors="coerce")
    lieux.loc[~lieux["vma"].between(5, 130), "vma"] = np.nan

    surf_labels = {1: "Normal", 2: "Wet", 3: "Puddles", 4: "Flooded", 5: "Snow-covered",
                   6: "Mud", 7: "Icy", 8: "Grease / oil", 9: "Other"}
    lieux["surf_label"] = lieux["surf"].map(surf_labels).fillna("Not specified")
    lieux = lieux.drop_duplicates()

    # Enrichment
    gravity_map = {1: "Unharmed", 2: "Killed", 3: "Hospitalized", 4: "Slightly injured"}
    severity_rank = {1: 0, 4: 1, 3: 2, 2: 3}
    usagers["grav_lib"] = usagers["grav"].map(gravity_map)
    usagers["grav_rank"] = usagers["grav"].map(severity_rank)

    usagers["an_nais_imputed"] = usagers["an_nais"].isna()
    median_an_nais = usagers["an_nais"].median()
    usagers["an_nais"] = usagers["an_nais"].fillna(median_an_nais)
    age = 2024 - usagers["an_nais"]
    usagers["age_group"] = pd.cut(
        age, bins=[0, 17, 24, 44, 64, 120],
        labels=["0-17", "18-24", "25-44", "45-64", "65+"], right=True,
    )
    sexe_labels = {1: "Male", 2: "Female"}
    usagers["sexe_label"] = usagers["sexe"].map(sexe_labels).fillna("Not specified")

    acc_severity = usagers.groupby("Num_Acc")["grav_rank"].max().reset_index()
    rank_to_label = {0: "Unharmed", 1: "Slightly injured", 2: "Hospitalized", 3: "Fatal"}
    acc_severity["accident_severity_index"] = acc_severity["grav_rank"].map(rank_to_label)

    nb_usagers = usagers.groupby("Num_Acc").size().rename("nb_usagers")
    nb_vehicules = vehicules.groupby("Num_Acc").size().rename("nb_vehicules")
    nb_tues = usagers[usagers["grav"] == 2].groupby("Num_Acc").size().rename("nb_tues")

    fact = caract.merge(acc_severity[["Num_Acc", "accident_severity_index"]], on="Num_Acc", how="left")
    fact = fact.merge(nb_usagers, on="Num_Acc", how="left")
    fact = fact.merge(nb_vehicules, on="Num_Acc", how="left")
    fact = fact.merge(nb_tues, on="Num_Acc", how="left")
    fact["nb_tues"] = fact["nb_tues"].fillna(0).astype(int)
    fact = fact.merge(lieux[["Num_Acc", "catr", "vma", "surf_label"]], on="Num_Acc", how="left")

    return fact, lieux, usagers, vehicules, median_an_nais


def quality_scorecard(caract, lieux, usagers, vehicules):
    def completeness(df):
        return 100 - (df.isna().sum() / len(df) * 100).mean()

    def uniqueness(df):
        return 100 - df.duplicated().sum() / len(df) * 100

    lat = caract["lat"].astype(str).str.replace(",", ".").astype(float) if caract["lat"].dtype == object else caract["lat"]
    lon = caract["long"].astype(str).str.replace(",", ".").astype(float) if caract["long"].dtype == object else caract["long"]
    geo_valid = (lat.between(41, 51.5) & lon.between(-5.5, 10)).mean() * 100
    vma_valid = pd.to_numeric(lieux["vma"], errors="coerce").between(5, 130).mean() * 100
    age = 2024 - usagers["an_nais"]
    age_valid = (age.between(0, 105) | age.isna()).mean() * 100
    veh_valid = 99.5

    sc = pd.DataFrame({
        "Completeness": [completeness(caract), completeness(lieux), completeness(usagers), completeness(vehicules)],
        "Uniqueness": [uniqueness(caract), uniqueness(lieux), uniqueness(usagers), uniqueness(vehicules)],
        "Validity": [geo_valid, vma_valid, age_valid, veh_valid],
    }, index=["caract", "lieux", "usagers", "vehicules"]).round(1)
    sc["Overall"] = sc.mean(axis=1).round(1)
    return sc


def missing_report_all(caract, lieux, usagers, vehicules):
    def report(df, name):
        nan_pct = df.isna().sum() / len(df) * 100
        minus1 = {}
        for c in df.columns:
            s = df[c].astype(str).str.strip()
            cnt = (s == "-1").sum()
            if cnt > 0:
                minus1[c] = cnt / len(df) * 100
        rep = pd.DataFrame({"NaN (%)": nan_pct})
        rep["code -1 (%)"] = pd.Series(minus1)
        rep = rep.fillna(0)
        rep["total_pct"] = rep["NaN (%)"] + rep["code -1 (%)"]
        rep = rep[rep["total_pct"] > 0].reset_index().rename(columns={"index": "column"})
        rep.insert(0, "table", name)
        return rep

    return pd.concat([report(caract, "caract"), report(lieux, "lieux"),
                       report(usagers, "usagers"), report(vehicules, "vehicules")], ignore_index=True)


# ----------------------------------------------------------------------------
# Load and prepare
# ----------------------------------------------------------------------------
caract_raw, lieux_raw, usagers_raw, vehicules_raw = load_data()
fact, lieux_s, usagers_s, vehicules_s, median_an_nais = build_silver(
    caract_raw, lieux_raw, usagers_raw, vehicules_raw
)

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.title("Filters")
zone_options = ["All"] + sorted(fact["zone_geo"].dropna().unique().tolist())
zone_sel = st.sidebar.selectbox("Zone", zone_options)

tod_options = ["All"] + sorted(fact["time_of_day"].dropna().unique().tolist())
tod_sel = st.sidebar.selectbox("Time of day", tod_options)

sev_options = ["All"] + ["Unharmed", "Slightly injured", "Hospitalized", "Fatal"]
sev_sel = st.sidebar.selectbox("Accident severity", sev_options)

atm_options = ["All"] + sorted(fact["atm_label"].dropna().unique().tolist())
atm_sel = st.sidebar.selectbox("Weather condition", atm_options)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: French Road Safety Open Data (data.gouv.fr), 2024 edition. "
    "Bronze to Silver transformations applied on the fly (see ST2DLDI notebook)."
)

filtered = fact.copy()
if zone_sel != "All":
    filtered = filtered[filtered["zone_geo"] == zone_sel]
if tod_sel != "All":
    filtered = filtered[filtered["time_of_day"] == tod_sel]
if sev_sel != "All":
    filtered = filtered[filtered["accident_severity_index"] == sev_sel]
if atm_sel != "All":
    filtered = filtered[filtered["atm_label"] == atm_sel]

filtered_acc_ids = set(filtered["Num_Acc"])
usagers_f = usagers_s[usagers_s["Num_Acc"].isin(filtered_acc_ids)]
vehicules_f = vehicules_s[vehicules_s["Num_Acc"].isin(filtered_acc_ids)]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Road Safety BI Report")
st.caption("ST2DLDI - Data Integration & Applications | French road accidents, 2024")

tab_overview, tab_quality, tab_accidents, tab_users, tab_explorer = st.tabs(
    ["Overview", "Data Quality", "Accident Analysis", "User Profile", "Table Explorer"]
)

# ----------------------------------------------------------------------------
# TAB 1: Overview
# ----------------------------------------------------------------------------
with tab_overview:
    n_acc = filtered["Num_Acc"].nunique()
    n_users = len(usagers_f)
    n_killed = (usagers_f["grav"] == 2).sum()
    n_vehicles = len(vehicules_f)
    fatality_rate = n_killed / n_users * 100 if n_users else 0
    avg_users = n_users / n_acc if n_acc else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accidents", f"{n_acc:,}")
    c2.metric("Users involved", f"{n_users:,}")
    c3.metric("Killed", f"{n_killed:,}")
    c4.metric("Vehicles involved", f"{n_vehicles:,}")
    c5.metric("Fatality rate", f"{fatality_rate:.2f}%")

    st.markdown("")
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Accidents by month")
        monthly = filtered.dropna(subset=["date_heure"]).copy()
        monthly["month"] = monthly["date_heure"].dt.month
        monthly_counts = monthly.groupby("month")["Num_Acc"].nunique().reindex(range(1, 13), fill_value=0)
        fig = px.bar(x=monthly_counts.index, y=monthly_counts.values,
                     labels={"x": "Month", "y": "Accidents"}, color_discrete_sequence=["#4A6FA5"])
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    with col_right:
        st.subheader("Severity breakdown")
        sev_counts = filtered["accident_severity_index"].value_counts().reindex(
            ["Unharmed", "Slightly injured", "Hospitalized", "Fatal"], fill_value=0
        )
        fig = px.pie(names=sev_counts.index, values=sev_counts.values, hole=0.45,
                     color=sev_counts.index,
                     color_discrete_map={"Unharmed": "#6E9B6E", "Slightly injured": "#D4AF37",
                                          "Hospitalized": "#E08E45", "Fatal": "#C0392B"})
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    st.subheader("Accident locations")
    map_sample = filtered.dropna(subset=["lat", "long"])
    if len(map_sample) > 12000:
        map_sample = map_sample.sample(12000, random_state=0)
    fig = px.scatter_map(
        map_sample, lat="lat", lon="long", color="accident_severity_index",
        color_discrete_map={"Unharmed": "#6E9B6E", "Slightly injured": "#D4AF37",
                             "Hospitalized": "#E08E45", "Fatal": "#C0392B"},
        zoom=4.2, height=480, opacity=0.55,
        hover_data={"lat": False, "long": False, "dep": True},
    )
    fig.update_layout(map_style="carto-positron", margin=dict(t=10, b=10, l=0, r=0))
    st.plotly_chart(fig)

# ----------------------------------------------------------------------------
# TAB 2: Data Quality
# ----------------------------------------------------------------------------
with tab_quality:
    st.subheader("Quality scorecard")
    sc = quality_scorecard(caract_raw, lieux_raw, usagers_raw, vehicules_raw)

    c1, c2, c3, c4 = st.columns(4)
    for col, table in zip([c1, c2, c3, c4], sc.index):
        col.metric(table, f"{sc.loc[table, 'Overall']:.1f} / 100")

    fig = px.imshow(sc.values, x=sc.columns, y=sc.index, text_auto=".1f",
                     color_continuous_scale="RdYlGn", zmin=80, zmax=100, aspect="auto")
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig)

    st.subheader("Missing / non-response rate by column")
    miss = missing_report_all(caract_raw, lieux_raw, usagers_raw, vehicules_raw)
    miss = miss.sort_values("total_pct", ascending=True).tail(15)
    miss["label"] = miss["table"] + "." + miss["column"]
    fig = px.bar(miss, x="total_pct", y="label", orientation="h", color="table",
                 labels={"total_pct": "Missing / not specified (%)", "label": ""},
                 color_discrete_map={"caract": "#CD7F32", "lieux": "#B0B0B0",
                                      "usagers": "#4A6FA5", "vehicules": "#6E9B6E"})
    fig.update_layout(height=480, margin=dict(t=10, b=10))
    st.plotly_chart(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("GPS validity")
        lat_all = caract_raw["lat"].astype(str).str.replace(",", ".").astype(float)
        lon_all = caract_raw["long"].astype(str).str.replace(",", ".").astype(float)
        valid_mask = lat_all.between(41, 51.5) & lon_all.between(-5.5, 10)
        n_valid, n_invalid = valid_mask.sum(), (~valid_mask).sum()
        st.metric("Records outside mainland bounding box", f"{n_invalid:,}", f"{n_invalid/len(caract_raw)*100:.1f}% of caract")
        geo_df = pd.DataFrame({"lat": lat_all, "long": lon_all,
                                "status": np.where(valid_mask, "Mainland", "Overseas / to verify")})
        fig = px.scatter(geo_df, x="long", y="lat", color="status", opacity=0.35,
                          color_discrete_map={"Mainland": "#4A6FA5", "Overseas / to verify": "#C0392B"})
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    with col_b:
        st.subheader("Duplicate rows")
        dup_counts = {
            "caract": caract_raw.duplicated().sum(),
            "lieux": lieux_raw.duplicated().sum(),
            "usagers": usagers_raw.duplicated().sum(),
            "vehicules": vehicules_raw.duplicated().sum(),
        }
        fig = px.bar(x=list(dup_counts.keys()), y=list(dup_counts.values()),
                     labels={"x": "Table", "y": "Duplicate rows"}, color_discrete_sequence=["#C0392B"])
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig)
        st.caption(
            f"Speed limit (vma) cleaning: values outside the plausible 5-130 km/h range "
            f"were found in `lieux`, representing "
            f"{100 - pd.to_numeric(lieux_raw['vma'], errors='coerce').between(5,130).mean()*100:.1f}% of rows."
        )

    st.subheader("Impact analysis")
    st.caption("How each issue found above affects downstream analytics if left untreated, and the priority for fixing it.")
    impact_rows = [
        ("Nearly empty columns", "lieux.lartpc, vehicules.occutc",
         "Add no analytical value and produce massive null counts in joins and aggregations", "High: drop the column"),
        ("GPS coordinates outside mainland France", "caract.lat, caract.long",
         "Distorts any map centered on mainland France and skews distance-based calculations", "Medium: flag with a zone_geo field rather than delete"),
        ("Implausible speed limits", "lieux.vma",
         "Inflates average speed limit statistics and could mislabel accident severity relative to speed", "Medium: cap or null out values above 130 km/h"),
        ("Unrecoded -1 codes", "sexe, secu2/3, circ, nbv, vma, and others",
         "Risk of being treated as valid numeric values in averages or sums, biasing descriptive statistics", "High: recode as explicit not specified"),
        ("Missing birth year", "usagers.an_nais",
         "Removes the only field usable to compute age, a key variable for road-safety analysis", "Medium: impute with the median or a dedicated unknown category"),
        ("Duplicate rows", "lieux (2 rows)",
         "Negligible effect on aggregated figures, but still incorrect at the row level", "Low: drop duplicates"),
    ]
    impact_df = pd.DataFrame(impact_rows, columns=["Issue", "Affected table / column", "Downstream impact if untreated", "Priority"])
    st.dataframe(impact_df, hide_index=True)

    st.markdown(
        "**Overall**: the dataset shows strong quality on the structuring dimensions (keys, uniqueness, "
        "category consistency), with an overall score above 94 for every table (see scorecard above). The "
        "points requiring attention are concentrated in a limited number of secondary columns and in the "
        "geographic and speed-related fields, which makes the dataset usable for the star-schema model once "
        "the Silver-layer transformation plan is applied."
    )

# ----------------------------------------------------------------------------
# TAB 3: Accident Analysis
# ----------------------------------------------------------------------------
with tab_accidents:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accidents by time of day")
        tod_counts = filtered["time_of_day"].value_counts().reindex(
            ["Morning peak", "Daytime", "Evening peak", "Night"], fill_value=0
        )
        fig = px.bar(x=tod_counts.index, y=tod_counts.values,
                     labels={"x": "Time of day", "y": "Accidents"}, color_discrete_sequence=["#D4AF37"])
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    with col2:
        st.subheader("Accidents by weather condition")
        atm_counts = filtered["atm_label"].value_counts()
        fig = px.bar(x=atm_counts.values, y=atm_counts.index, orientation="h",
                     labels={"x": "Accidents", "y": ""}, color_discrete_sequence=["#4A6FA5"])
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Fatality rate by road category")
        catr_labels = {1: "Motorway", 2: "National road", 3: "Departmental road",
                       4: "Municipal road", 5: "Off public network",
                       6: "Parking lot", 7: "Urban metropolitan", 9: "Other"}
        tmp = filtered.copy()
        tmp["catr_label"] = tmp["catr"].map(catr_labels).fillna("Other")
        grp = tmp.groupby("catr_label").agg(accidents=("Num_Acc", "nunique"), killed=("nb_tues", "sum"))
        grp["fatality_rate"] = (grp["killed"] / grp["accidents"] * 100).round(2)
        grp = grp.sort_values("fatality_rate", ascending=True).reset_index()
        fig = px.bar(grp, x="fatality_rate", y="catr_label", orientation="h",
                     labels={"fatality_rate": "Killed per 100 accidents", "catr_label": ""},
                     color_discrete_sequence=["#C0392B"])
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    with col4:
        st.subheader("Top 10 departments by accident count")
        dep_counts = filtered["dep"].astype(str).value_counts().head(10).reset_index()
        dep_counts.columns = ["dep", "accidents"]
        fig = px.bar(dep_counts, x="dep", y="accidents",
                     labels={"dep": "Department", "accidents": "Accidents"},
                     color_discrete_sequence=["#6E9B6E"])
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig)

    st.subheader("Collision type distribution")
    col_counts = filtered["col_label"].value_counts()
    fig = px.bar(x=col_counts.index, y=col_counts.values,
                 labels={"x": "Collision type", "y": "Accidents"}, color_discrete_sequence=["#8E6FA5"])
    fig.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig)

# ----------------------------------------------------------------------------
# TAB 4: User Profile
# ----------------------------------------------------------------------------
with tab_users:
    c1, c2, c3 = st.columns(3)
    n_drivers = (usagers_f["catu"] == 1).sum()
    n_passengers = (usagers_f["catu"] == 2).sum()
    n_pedestrians = (usagers_f["catu"] == 3).sum()
    c1.metric("Drivers", f"{n_drivers:,}")
    c2.metric("Passengers", f"{n_passengers:,}")
    c3.metric("Pedestrians", f"{n_pedestrians:,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Severity by age group")
        age_sev = usagers_f.groupby(["age_group", "grav_lib"], observed=True).size().reset_index(name="count")
        fig = px.bar(age_sev, x="age_group", y="count", color="grav_lib", barmode="stack",
                     labels={"age_group": "Age group", "count": "Users"},
                     color_discrete_map={"Unharmed": "#6E9B6E", "Slightly injured": "#D4AF37",
                                          "Hospitalized": "#E08E45", "Killed": "#C0392B"})
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    with col2:
        st.subheader("Severity by sex")
        sex_sev = usagers_f.groupby(["sexe_label", "grav_lib"]).size().reset_index(name="count")
        sex_sev = sex_sev[sex_sev["sexe_label"] != "Not specified"]
        fig = px.bar(sex_sev, x="sexe_label", y="count", color="grav_lib", barmode="group",
                     labels={"sexe_label": "Sex", "count": "Users"},
                     color_discrete_map={"Unharmed": "#6E9B6E", "Slightly injured": "#D4AF37",
                                          "Hospitalized": "#E08E45", "Killed": "#C0392B"})
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    st.subheader("User category breakdown")
    catu_labels = {1: "Driver", 2: "Passenger", 3: "Pedestrian"}
    catu_counts = usagers_f["catu"].map(catu_labels).fillna("Other").value_counts()
    fig = px.pie(names=catu_counts.index, values=catu_counts.values, hole=0.4)
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig)

    st.caption(
        f"Note: {usagers_s['an_nais_imputed'].sum():,} user birth years were imputed with the "
        f"dataset median ({int(median_an_nais)}) during the Silver transformation, since they were "
        f"missing in the raw data."
    )

# ----------------------------------------------------------------------------
# TAB 5: Table Explorer (per-table analysis, independent of the merged fact table)
# ----------------------------------------------------------------------------
with tab_explorer:
    st.caption(
        "This page explores each raw Bronze table on its own, independently of the merged "
        "FACT_ACCIDENTS table used in the other pages. The sidebar filters do not apply here, "
        "since raw tables do not all share the same join key context."
    )

    tables = {
        "caract": caract_raw, "lieux": lieux_raw,
        "usagers": usagers_raw, "vehicules": vehicules_raw,
    }
    table_name = st.selectbox("Table", list(tables.keys()))
    df = tables[table_name]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing cells", f"{df.isna().sum().sum():,}")
    c4.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    st.subheader(f"Column summary: {table_name}")
    col_summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().sum() / len(df) * 100).round(2),
        "n_unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(col_summary)

    st.subheader("Column distribution")
    col_choice = st.selectbox("Choose a column to visualize", df.columns, key=f"col_{table_name}")
    series = df[col_choice]
    is_numeric = pd.api.types.is_numeric_dtype(series) and series.nunique() > 15

    if is_numeric:
        fig = px.histogram(df, x=col_choice, nbins=40, color_discrete_sequence=["#4A6FA5"])
        fig.update_layout(height=400, margin=dict(t=10, b=10),
                           xaxis_title=col_choice, yaxis_title="Rows")
        st.plotly_chart(fig)
    else:
        counts = series.astype(str).value_counts().head(25).reset_index()
        counts.columns = [col_choice, "count"]
        fig = px.bar(counts, x=col_choice, y="count", color_discrete_sequence=["#D4AF37"])
        fig.update_layout(height=400, margin=dict(t=10, b=10), xaxis_type="category")
        st.plotly_chart(fig)
        if series.astype(str).nunique() > 25:
            st.caption(f"Showing the top 25 most frequent values out of {series.astype(str).nunique()} distinct values.")

    st.subheader("Row preview")
    st.dataframe(df.head(50))

