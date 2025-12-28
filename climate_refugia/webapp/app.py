from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def build_map(
    base_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    future_df: pd.DataFrame,
    show_future: bool,
) -> folium.Map:
    if base_df.empty:
        center = [0, 0]
    else:
        center = [base_df["lat"].mean(), base_df["lon"].mean()]

    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    if not base_df.empty:
        heat_points = base_df[["lat", "lon"]].dropna().values.tolist()
        HeatMap(heat_points, radius=12, blur=8, min_opacity=0.3).add_to(fmap)

    if not clusters_df.empty:
        for _, row in clusters_df.iterrows():
            color = "green" if row.get("is_refugia", False) else "orange"
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Cluster {row['cluster_id']} | Individuals {row['num_individuals']}",
            ).add_to(fmap)

    if show_future and not future_df.empty:
        future_points = future_df[future_df["is_refugia_pred"]]
        for _, row in future_points.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.6,
                popup=f"{row['species']} | Prob {row['refugia_probability']:.2f}",
            ).add_to(fmap)

    return fmap


def main() -> None:
    st.set_page_config(page_title="Climate Refugia Explorer", layout="wide")
    st.title("Climate Refugia Explorer")

    outputs_dir = Path(
        st.sidebar.text_input(
            "Outputs directory",
            value=str(Path(__file__).resolve().parents[2] / "outputs"),
        )
    )
    aligned_df = load_parquet(outputs_dir / "aligned_data.parquet")
    heat_df = load_parquet(outputs_dir / "heat_events.parquet")
    clusters_df = load_parquet(outputs_dir / "refugia_clusters.parquet")

    future_files = sorted(outputs_dir.glob("future_refugia_*.parquet"))
    scenarios = [path.stem.replace("future_refugia_", "") for path in future_files]

    st.sidebar.subheader("Filters")
    species_options = sorted(heat_df["species"].unique()) if not heat_df.empty else []
    selected_species = st.sidebar.multiselect("Species", species_options, default=species_options)

    show_future = st.sidebar.checkbox("Show future refugia", value=False)
    future_df = pd.DataFrame()
    if show_future and scenarios:
        selected_scenario = st.sidebar.selectbox("Scenario", scenarios, index=0)
        future_path = outputs_dir / f"future_refugia_{selected_scenario}.parquet"
        future_df = load_parquet(future_path)

    if selected_species:
        heat_df = heat_df[heat_df["species"].isin(selected_species)]
        if "dominant_species" in clusters_df.columns:
            clusters_df = clusters_df[clusters_df["dominant_species"].isin(selected_species)]
        if not future_df.empty:
            future_df = future_df[future_df["species"].isin(selected_species)]

    st.subheader("Refugia Map")
    fmap = build_map(heat_df, clusters_df, future_df, show_future)
    st.components.v1.html(fmap._repr_html_(), height=600, scrolling=False)

    st.subheader("Heat Event Summary")
    if heat_df.empty:
        st.info("No heat events detected.")
    else:
        summary = heat_df.groupby("species").agg(
            points=("heat_event_id", "count"),
            events=("heat_event_id", lambda x: x.nunique()),
            mean_temp=("temp_c", "mean"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True)

    st.subheader("Cluster Details")
    if clusters_df.empty:
        st.info("No refugia clusters available.")
    else:
        st.dataframe(clusters_df, use_container_width=True)

    if show_future:
        st.subheader("Future Refugia Predictions")
        if future_df.empty:
            st.info("No future refugia predictions available.")
        else:
            st.dataframe(future_df.head(500), use_container_width=True)


if __name__ == "__main__":
    main()
