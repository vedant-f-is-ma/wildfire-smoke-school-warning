# Streamlit GIS app for wildfire-smoke school warnings.
# Loads station-level predictions, aggregates to CA counties, and renders a date-driven choropleth with optional station overlays.

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import folium
import branca.colormap as cm
import streamlit as st
from streamlit_folium import st_folium

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="School Smoke Risk GIS", layout="wide")

DEFAULT_CENTER = (37.55, -121.99)  # Fremont-ish
DEFAULT_ZOOM = 6

COUNTIES_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
BULLETIN_CSV = "school_smoke_bulletin_station_level.csv"  # put next to app.py

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_bulletin(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize expected cols
    df["Date Local"] = pd.to_datetime(df["Date Local"]).dt.date
    for c in ["Latitude", "Longitude", "risk_prob", "warn_flag"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in bulletin CSV: {c}")

    df = df.dropna(subset=["Latitude", "Longitude", "risk_prob"]).copy()
    df["risk_prob"] = pd.to_numeric(df["risk_prob"], errors="coerce").clip(0, 1)
    df["warn_flag"] = pd.to_numeric(df["warn_flag"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_ca_counties() -> gpd.GeoDataFrame:
    counties = gpd.read_file(COUNTIES_GEOJSON_URL)
    # Filter to California (FIPS starts with "06")
    counties["fips"] = counties["id"].astype(str).str.zfill(5)
    counties = counties[counties["fips"].str.startswith("06")].copy()
    counties = counties.set_crs(epsg=4326, allow_override=True)
    return counties[["fips", "geometry"]]

@st.cache_data(show_spinner=False)
def spatial_join_points_to_counties(df_points: pd.DataFrame, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_points = gpd.GeoDataFrame(
        df_points,
        geometry=[Point(xy) for xy in zip(df_points["Longitude"], df_points["Latitude"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf_points, counties, how="left", predicate="within")
    joined = joined.dropna(subset=["fips"]).copy()
    return joined

def aggregate_county_daily(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    # One row per county per day with metrics
    out = (joined.groupby(["Date Local", "fips"], as_index=False)
           .agg(
               max_risk=("risk_prob", "max"),
               avg_risk=("risk_prob", "mean"),
               num_sites=("risk_prob", "count"),
               num_warned=("warn_flag", "sum"),
           ))
    return out

def render_map(counties_gdf: gpd.GeoDataFrame,
               county_daily: pd.DataFrame,
               stations_day: pd.DataFrame,
               selected_date,
               metric: str,
               show_stations: bool,
               center=DEFAULT_CENTER,
               zoom=DEFAULT_ZOOM):

    # Merge metrics onto county geometries for the day
    day = county_daily[county_daily["Date Local"] == selected_date].copy()
    day_geo = counties_gdf.merge(day, on="fips", how="left")

    # Fill NAs: counties with no monitors that day get 0
    for col in ["max_risk", "avg_risk", "num_warned", "num_sites"]:
        if col in day_geo.columns:
            day_geo[col] = day_geo[col].fillna(0)

    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")

    vmin = float(day_geo[metric].min())
    vmax = float(day_geo[metric].max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = f"{metric} on {selected_date}"
    colormap.add_to(m)

    def style_fn(feat):
        val = feat["properties"].get(metric, 0) or 0
        return {
            "fillOpacity": 0.65,
            "weight": 0.6,
            "fillColor": colormap(val),
        }

    tooltip_fields = ["fips", metric, "num_warned", "num_sites"]
    tooltip_aliases = ["County FIPS", metric, "Warned sites", "Total sites"]

    folium.GeoJson(
        day_geo,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True),
        name="CA Counties"
    ).add_to(m)

    # Optional station overlay
    if show_stations and stations_day is not None and len(stations_day):
        for _, r in stations_day.iterrows():
            # Size by risk; keep it sane
            radius = float(2 + 10 * np.clip(r["risk_prob"], 0, 1))
            folium.CircleMarker(
                location=[r["Latitude"], r["Longitude"]],
                radius=radius,
                fill=True,
                fill_opacity=0.8,
                weight=1,
                popup=f"risk={r['risk_prob']:.2f} warn={int(r['warn_flag'])}"
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# -----------------------------
# App UI
# -----------------------------
st.title("School Smoke Risk GIS")
st.caption("County-level choropleth + optional station overlay. Built for school decision support (tomorrow warning).")

with st.sidebar:
    st.header("Controls")
    st.write("Upload a bulletin CSV or use the default next to app.py.")
    uploaded = st.file_uploader("Bulletin CSV", type=["csv"])
    show_stations = st.toggle("Show station points", value=True)

    zoom_local = st.toggle("Zoom to Bay Area", value=False)
    metric = st.selectbox("Metric", ["max_risk", "avg_risk", "num_warned"], index=0)

# Load data
if uploaded is not None:
    df_bulletin = load_bulletin(uploaded)
else:
    df_bulletin = load_bulletin(BULLETIN_CSV)

counties = load_ca_counties()
joined = spatial_join_points_to_counties(df_bulletin, counties)
county_daily = aggregate_county_daily(joined)

dates = sorted(county_daily["Date Local"].unique())
if not dates:
    st.error("No dates available after spatial join. Check that your stations are in CA and have valid lat/lon.")
    st.stop()

selected_date = st.selectbox("Select date", dates, index=len(dates)-1)

# Station overlay table for that day
stations_day = df_bulletin[df_bulletin["Date Local"] == selected_date].copy()

# Center/zoom choice
if zoom_local:
    center = DEFAULT_CENTER
    zoom = 9
else:
    center = (37.0, -119.5)  # CA-ish
    zoom = 6

# Render map
m = render_map(
    counties_gdf=counties,
    county_daily=county_daily,
    stations_day=stations_day,
    selected_date=selected_date,
    metric=metric,
    show_stations=show_stations,
    center=center,
    zoom=zoom
)

col1, col2 = st.columns([2, 1])

with col1:
    st_folium(m, height=650, width=None)

with col2:
    st.subheader("Day summary")
    day = county_daily[county_daily["Date Local"] == selected_date].copy()
    st.metric("Counties with any warnings", int((day["num_warned"] > 0).sum()))
    st.metric("Total warned sites", int(day["num_warned"].sum()))
    st.metric("Max county risk", float(day["max_risk"].max()) if len(day) else 0.0)

    st.subheader("Top stations by risk")
    top = stations_day.sort_values("risk_prob", ascending=False).head(15)
    show_cols = [c for c in ["County Name","Latitude","Longitude","risk_prob","warn_flag","top_reasons"] if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True)
