"""
This Streamlit app provides a minimal web interface to run the Biodiversity Mapping Pipeline.
It currently allows you to:
- Enter the YOLO model path, input image directory, and output directory.
- Run the same processing steps as the terminal version.
- See logs and the final heatmap in the browser.

Note:
This script does not add any extra functionality beyond replacing the terminal CLI.
It is intended as a simple alternative to command-line execution.
"""
import streamlit as st
import pandas as pd
import os
from pipeline import run_pipeline
from biodiversity import deduplicate_points
import plotly.express as px
from pyproj import Transformer
import plotly.graph_objects as go

START_THRESHOLD = 0.5

st.title("🌿 Biodiversity mapping pipeline")

model_path = st.text_input("YOLO model path", "/path/to/best.pt")

image_dir = st.text_input("Path to input images directory", "examples/small_drone")

mapbox_token = st.text_input(
    "Optional mapbox access token (for satellite imagery). Leave blank to use OpenStreetMap. The token is free, but an account is needed.", 
    "pk.eyJ1IjoicmFiZWFpZmVhbnlpIiwiYSI6ImNtZDFjcHY5eTEzZXUya3FuMjhiOW1scXQifQ.5o4YFlUNULNSU9V-JzGcRQ"
)

robot_tick = st.checkbox("Robot data")
threshold = st.number_input("Deduplication distance threshold (m)", value=START_THRESHOLD, min_value=0.0, step=0.1)

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Run Pipeline"):
    if not os.path.exists(model_path):
        st.error("Model path does not exist.")
    elif not os.path.exists(image_dir):
        st.error("Image directory does not exist.")
    else:
        st.info("Running pipeline... please wait.")
        if not robot_tick:
            fig, logs, stats, global_pos = run_pipeline(model_path, image_dir, "drone")
        else:
            fig, logs, stats, global_pos = run_pipeline(model_path, image_dir)

        st.session_state.results = (fig, logs, stats, global_pos)

if st.session_state.results:
    fig, logs, stats, global_pos = st.session_state.results
    if logs:
        with st.expander("Log output", expanded=False):
            st.text("\n".join(logs))

    if stats:
        if "detections_per_class" in stats:
            with st.expander("Summary of detections per class", expanded=False):
                df_class = pd.DataFrame(
                    [{"Class": k, "Count": v} for k, v in stats["detections_per_class"].items()]
                )
                st.table(df_class.style.hide(axis="index"))

        if "detections_per_image_distribution" in stats:
            with st.expander("Distribution of detections per image", expanded=False):
                df_dist = pd.DataFrame(
                    [{"Detections": k, "Number of Images": v} for k, v in sorted(stats["detections_per_image_distribution"].items())]
                )
                st.table(df_dist.style.hide(axis="index"))
                
        if global_pos:
            global_pos = deduplicate_points(global_pos, distance_threshold=threshold)

            df = pd.DataFrame(global_pos, columns=["X", "Y", "Class", "Confidence"])

            transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

            if robot_tick:
                df[["lon", "lat"]] = df.apply(
                    lambda row: pd.Series(transformer.transform(row["X"], row["Y"])),
                    axis=1
                )
            else:
                df[["lat", "lon"]] = df.apply(
                    lambda row: pd.Series(transformer.transform(row["X"], row["Y"])),
                    axis=1
                )

            with st.expander("Longitude and latitude", expanded=False):
                st.write(df[["X", "Y", "lon", "lat"]])

            with st.expander("Filter species to display", expanded=True):
                unique_classes = sorted(df["Class"].unique())
                selected_classes = st.multiselect(
                    "Select species to show",
                    options=unique_classes,
                    default=unique_classes
                )

            df_filtered = df[df["Class"].isin(selected_classes)]

            fig = px.scatter_mapbox(
                df_filtered,
                lat="lat",
                lon="lon",
                hover_name="Class",
                hover_data=["Confidence"],
                color="Class",
                zoom=1,
                height=700
            )

            if mapbox_token:
                fig.update_layout(

                    mapbox=dict(
                        style="satellite",
                        accesstoken=mapbox_token,
                        center=go.layout.mapbox.Center(
                            lat=df["lat"].mean(),
                            lon=df["lon"].mean()
                        ),
                        pitch=0,
                        zoom=19
                    ),
                    margin={"r":0,"t":0,"l":0,"b":0},
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )
            
            else:
                fig.update_layout(
                    mapbox=dict(
                        style="open-street-map",
                        center=go.layout.mapbox.Center(
                            lat=df["lat"].mean(),
                            lon=df["lon"].mean()
                        ),
                        pitch=0,
                        zoom=18
                    ),
                    margin={"r":0,"t":0,"l":0,"b":0},
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )

            st.plotly_chart(fig, use_container_width=True)
