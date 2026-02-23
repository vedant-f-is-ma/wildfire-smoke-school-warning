# app.py
# Streamlit GIS app: date/metric dropdown + CA county choropleth + optional station overlay for your school smoke warnings.
# Designed to be cache-safe (no unhashable GeoDataFrame args in cached functions) and deployment-friendly.

from __future__ import annotations

import io
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# Geospatial stack
import geopandas as gpd
from shapely.geometry import Point

# Mapping stack
import folium
import branca.colormap as cm
from streamlit_folium import st_folium


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="School Smoke Risk GIS", layout="wide")

DEFAULT_CENTER_CA = (37.0, -119.5)          # California-ish
DEFAULT_CENTER_BAY = (37.55, -121.99)        # Fremont-ish
DEFAULT_ZOOM_CA = 6
DEFAULT_ZOOM_BAY = 9

COUNTIES_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

DEFAULT_BULLETIN_CSV = "school_smoke_bulletin_station_level.csv"


# -----------------------------
# Cached loaders (safe to cache)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_bulletin_csv(file_like: Union[str, io.BytesIO]) -> pd.DataFrame:
    """
    Loads station-level bulletin CSV and normalizes columns.
    This is cached because the input is hashable (path or uploaded bytes).
    """
    df = pd.read_csv(file_like)

    # Normalize expected columns
    required = {"Date Local", "Latitude", "Longitude", "risk_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bulletin CSV is missing columns: {sorted(missing)}")

    df["Date Local"] = pd.to_datetime(df["Date Local"], errors="coerce").dt.date
    df = df.dropna(subset=["Date Local", "Latitude", "Longitude", "risk_prob"]).copy()

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["risk_prob"] = pd.to_numeric(df["risk_prob"], errors="coerce").clip(0, 1)

    # Optional columns
    if "warn_flag" in df.columns:
        df["warn_flag"] = pd.to_numeric(df["warn_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["warn_flag"] = 0

    # Helpful display columns (optional)
    for col in ["County Name", "top_reasons"]:
        if col not in df.columns:
            df[col] = ""

    df = df.dropna(subset=["Latitude", "Longitude", "risk_prob"]).copy()
    return df


@st.cache_resource(show_spinner=False)
def load_ca_counties() -> gpd.GeoDataFrame:
    """
    Loads CA counties geometry (FIPS prefix 06) as a GeoDataFrame.
    Cached as a resource because it’s essentially static geometry.
    """
    counties = gpd.read_file(COUNTIES_GEOJSON_URL)
    counties["fips"] = counties["id"].astype(str).str.zfill(5)
    counties = counties[counties["fips"].str.startswith("06")].copy()
    counties = counties.set_crs(epsg=4326, allow_override=True)
    return counties[["fips", "geometry"]]


# -----------------------------
# Non-cached geospatial compute (avoids Streamlit hashing issues)
# -----------------------------
def spatial_join_points_to_counties(df_points: pd.DataFrame, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatially joins station points to CA counties.
    Not cached (GeoDataFrame hashing is painful and the scale here is manageable).
    """
    gdf_points = gpd.GeoDataFrame(
        df_points.copy(),
        geometry=gpd.points_from_xy(df_points["Longitude"], df_points["Latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(gdf_points, counties, how="left", predicate="within")
    joined = joined.dropna(subset=["fips"]).copy()
    return joined


def aggregate_county_daily(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregates station outputs into county-by-date metrics.
    Output is a plain DataFrame (hashable-ish, easy to work with).
    """
    out = (
        joined.groupby(["Date Local", "fips"], as_index=False)
        .agg(
            max_risk=("risk_prob", "max"),
            avg_risk=("risk_prob", "mean"),
            num_sites=("risk_prob", "count"),
            num_warned=("warn_flag", "sum"),
        )
    )
    return out


def render_folium_map(
    counties_gdf: gpd.GeoDataFrame,
    county_daily: pd.DataFrame,
    stations_day: pd.DataFrame,
    selected_date,
    metric: str,
    show_stations: bool,
    zoom_bay: bool,
) -> folium.Map:
    """
    Renders a folium choropleth map + optional station overlay for a selected date.
    """
    center = DEFAULT_CENTER_BAY if zoom_bay else DEFAULT_CENTER_CA
    zoom = DEFAULT_ZOOM_BAY if zoom_bay else DEFAULT_ZOOM_CA

    # Filter to selected date and merge
    day = county_daily[county_daily["Date Local"] == selected_date].copy()
    day_geo = counties_gdf.merge(day, on="fips", how="left")

    # Counties with no stations that day get zeros
    for col in ["max_risk", "avg_risk", "num_warned", "num_sites"]:
        if col in day_geo.columns:
            day_geo[col] = day_geo[col].fillna(0)

    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")

    # Colormap
    vmin = float(day_geo[metric].min()) if len(day_geo) else 0.0
    vmax = float(day_geo[metric].max()) if len(day_geo) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = f"{metric} on {selected_date}"
    colormap.add_to(m)

    def style_fn(feat):
        val = feat["properties"].get(metric, 0) or 0
        return {"fillOpacity": 0.65, "weight": 0.6, "fillColor": colormap(val)}

    tooltip_fields = ["fips", metric, "num_warned", "num_sites"]
    tooltip_aliases = ["County FIPS", metric, "Warned sites", "Total sites"]

    folium.GeoJson(
        day_geo.to_json(),
        name="CA Counties",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True),
    ).add_to(m)

    # Optional station overlay
    if show_stations and stations_day is not None and len(stations_day):
        for _, r in stations_day.iterrows():
            radius = float(2 + 10 * np.clip(r["risk_prob"], 0, 1))
            popup_bits = [
                f"risk={float(r['risk_prob']):.2f}",
                f"warn={int(r.get('warn_flag', 0))}",
            ]
            if "County Name" in stations_day.columns and r.get("County Name", ""):
                popup_bits.insert(0, str(r["County Name"]))
            if "top_reasons" in stations_day.columns and r.get("top_reasons", ""):
                popup_bits.append(str(r["top_reasons"]))

            folium.CircleMarker(
                location=[float(r["Latitude"]), float(r["Longitude"])],
                radius=radius,
                fill=True,
                fill_opacity=0.8,
                weight=1,
                popup=" | ".join(popup_bits),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# -----------------------------
# UI
# -----------------------------
st.title("School Smoke Risk GIS")
st.caption("County choropleth + station overlay with date/metric controls. Built for operational decision support (schools).")

with st.sidebar:
    st.header("Controls")

    uploaded = st.file_uploader("Upload bulletin CSV", type=["csv"])
    st.divider()

    metric = st.selectbox("Metric", ["max_risk", "avg_risk", "num_warned"], index=0)
    show_stations = st.toggle("Show station points", value=True)
    zoom_bay = st.toggle("Zoom to Bay Area", value=True)

    st.divider()
    st.write("Tip: If deploys get cranky, use Python 3.11 in your platform settings. Geospatial wheels are happiest there.")

# Load data
try:
    if uploaded is not None:
        df_bulletin = load_bulletin_csv(uploaded)
    else:
        df_bulletin = load_bulletin_csv(DEFAULT_BULLETIN_CSV)
except Exception as e:
    st.error("Couldn’t load bulletin CSV. Confirm the file exists (or upload one) and has required columns.")
    st.exception(e)
    st.stop()

# Load counties
try:
    counties = load_ca_counties()
except Exception as e:
    st.error("Couldn’t load CA counties geometry. This is usually a geopandas/shapely install issue.")
    st.exception(e)
    st.stop()

# Compute joins + aggregates (not cached to avoid unhashable GeoDataFrame args)
try:
    joined = spatial_join_points_to_counties(df_bulletin, counties)
    county_daily = aggregate_county_daily(joined)
except Exception as e:
    st.error("Spatial processing failed (join/aggregate). Check that station coordinates are valid and within CA.")
    st.exception(e)
    st.stop()

dates = sorted(county_daily["Date Local"].unique())
if not dates:
    st.warning("No mappable dates found after spatial join. Are your stations in California (lat/lon) and not missing?")
    st.stop()

selected_date = st.selectbox("Select date", dates, index=len(dates) - 1)

# Station overlay for selected day
stations_day = df_bulletin[df_bulletin["Date Local"] == selected_date].copy()

# Render
m = render_folium_map(
    counties_gdf=counties,
    county_daily=county_daily,
    stations_day=stations_day,
    selected_date=selected_date,
    metric=metric,
    show_stations=show_stations,
    zoom_bay=zoom_bay,
)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st_folium(m, height=650, width=None)

with col2:
    st.subheader("Day summary")

    day = county_daily[county_daily["Date Local"] == selected_date].copy()
    counties_with_warn = int((day["num_warned"] > 0).sum()) if len(day) else 0
    total_warned_sites = int(day["num_warned"].sum()) if len(day) else 0
    max_county_risk = float(day["max_risk"].max()) if len(day) else 0.0

    st.metric("Counties with warnings", counties_with_warn)
    st.metric("Total warned sites", total_warned_sites)
    st.metric("Max county risk", round(max_county_risk, 3))

    st.subheader("Top stations by risk")
    top = stations_day.sort_values("risk_prob", ascending=False).head(15)
    cols = [c for c in ["County Name", "risk_prob", "warn_flag", "top_reasons", "Latitude", "Longitude"] if c in top.columns]
    st.dataframe(top[cols], use_container_width=True)

    with st.expander("Data QA"):
        st.write("Bulletin rows:", len(df_bulletin))
        st.write("Joined rows (in CA counties):", len(joined))
        st.write("Dates available:", len(dates))
