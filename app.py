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
import os
from pipeline import run_pipeline

st.title("🌿 Biodiversity Mapping Pipeline")

model_path = st.text_input("YOLO Model Path", "/path/to/best.pt")
image_dir = st.text_input("Input Images Directory", "/path/to/images")
output_dir = st.text_input("Output Directory", "/path/to/output")

if st.button("Run Pipeline"):
    if not os.path.exists(model_path):
        st.error("Model path does not exist.")
    elif not os.path.exists(image_dir):
        st.error("Image directory does not exist.")
    else:
        st.info("Running pipeline... please wait.")
        fig, logs = run_pipeline(model_path, image_dir, output_dir)

        if logs:
            st.subheader("Log Output")
            st.text("\n".join(logs))

        if fig:
            st.pyplot(fig)
            st.success("Pipeline finished!")
        else:
            st.warning("No data to plot.")
