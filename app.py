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

st.title("🌿 Biodiversity mapping pipeline")

model_path = st.text_input("YOLO model path", "/path/to/best.pt")
image_dir = st.text_input("Path to input images directory", "/path/to/images")

if st.button("Run Pipeline"):
    if not os.path.exists(model_path):
        st.error("Model path does not exist.")
    elif not os.path.exists(image_dir):
        st.error("Image directory does not exist.")
    else:
        st.info("Running pipeline... please wait.")
        fig, logs, stats = run_pipeline(model_path, image_dir)

        if logs:
            st.subheader("Log Output")
            st.text("\n".join(logs))

        if stats:
            if "detections_per_class" in stats:
                st.subheader("Summary of detections per class")
                df_class = pd.DataFrame(
                    [{"Class": k, "Count": v} for k, v in stats["detections_per_class"].items()]
                )
                st.table(df_class.style.hide(axis="index"))

            if "detections_per_image_distribution" in stats:
                st.subheader("Distribution of detections per image")
                df_dist = pd.DataFrame(
                    [{"Detections": k, "Number of Images": v} for k, v in sorted(stats["detections_per_image_distribution"].items())]
                )
                st.table(df_dist.style.hide(axis="index"))

        if fig:
            st.pyplot(fig)
            st.success("Pipeline finished!")
        else:
            st.warning("No data to plot.")
