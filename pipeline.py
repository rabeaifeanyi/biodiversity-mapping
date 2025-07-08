"""
Biodiversity Mapping Pipeline

This script implements a complete pipeline to:
- Run YOLO object detection on input images.
- Parse the resulting XML annotations.
- Transform detected bounding boxes into world coordinates.
- Cache the coordinates to avoid recomputation.
- Generate a heatmap of all detected plant positions.

Usage:
- From the terminal (CLI):
    python pipeline.py --model_path ... --image_dir ...
- From Streamlit or other Python code:
    from pipeline import run_pipeline
    fig, logs = run_pipeline(...)

Notes:
- This script requires a trained YOLO model and corresponding image metadata (JSON files).
- The pipeline outputs logs and a matplotlib figure.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from biodiversity import (
    run_yolo_prediction,
    xml_unpack,
    main_coordinates,
    load_cache,
    save_cache,
    CameraConfig,
    plot_map
)

def run_pipeline(model_path, image_dir):
    """
    Runs the full detection and mapping pipeline.

    Args:
        model_path (str): Path to the YOLO .pt weights file.
        image_dir (str): Directory containing input images.

    Returns:
        tuple:
            - fig (matplotlib.figure.Figure or None): Heatmap figure, or None if no data.
            - logs (list of str): Log messages describing the process.

    Steps performed:
        - Load or create a cache of coordinates.
        - Run YOLO predictions on input images.
        - Parse XML annotations.
        - Convert pixel positions to world coordinates.
        - Save updated cache.
        - Generate a heatmap plot.
    """
    logs = []

    def log(msg):
        logs.append(msg)

    log("Starting pipeline...")
    output_dir = Path(image_dir)
    cache_path = output_dir / "coordinates_cache.pkl"

    cache = load_cache(cache_path)
    log("Cache loaded.")

    if cache:
        log("Using existing cache, skipping YOLO prediction.")
        global_pos = []
        for coords in cache.values():
            global_pos.extend(coords)

        if not global_pos:
            log("Cache was empty, no coordinates to plot.")
            return None, logs

        fig = plot_map(global_pos, background_path=None)
        return fig, logs

    run_yolo_prediction(model_path, image_dir)
    log("YOLO prediction completed.")

    camera_cfg = CameraConfig()
    global_pos = []

    for file in os.listdir(output_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            basename = os.path.splitext(file)[0]
            xml_path = output_dir / f"{basename}.xml"

            if not xml_path.exists():
                log(f"{basename}: XML missing, skipping.")
                continue

            if file in cache:
                coords = cache[file]
                global_pos.extend(coords)
                log(f"Loaded cached coordinates for {file}.")
            else:
                objects, width, height = xml_unpack(xml_path)
                coords = []

                log(f"Processing {file}: {len(objects)} detections found.")
                for obj in objects:
                    bbox = obj['bbox']
                    x_pixel = (bbox['xmin'] + bbox['xmax']) / 2
                    y_pixel = (bbox['ymin'] + bbox['ymax']) / 2

                    result = main_coordinates(
                        x_pixel, y_pixel,
                        camera_cfg,
                        output_dir,
                        file,
                        width, height
                    )

                    # if result:
                    #     coords.append(result)
                    #     global_pos.append(result)

                    if result:
                        coords.append( (result[0], result[1], obj["name"], obj["confidence"]) )
                        global_pos.append( (result[0], result[1], obj["name"], obj["confidence"]) )


                cache[file] = coords

    save_cache(cache_path, cache)
    log(f"Saved cache with {len(global_pos)} total coordinates.")

    if not global_pos:
        log("No coordinates found - no heatmap.")
        return None, logs

    x_coords = np.array([p[0] for p in global_pos])
    y_coords = np.array([p[1] for p in global_pos])

    xy = np.vstack([x_coords, y_coords])
    kde = gaussian_kde(xy)

    x_grid, y_grid = np.mgrid[
        x_coords.min():x_coords.max():100j,
        y_coords.min():y_coords.max():100j
    ]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords).reshape(x_grid.shape)

    fig = plot_map(global_pos, background_path=None) #TODO
    return fig, logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biodiversity Mapping Pipeline")
    parser.add_argument("--model_path", type=str, default=os.path.join("model", "best.pt"))
    parser.add_argument("--image_dir", type=str, default=os.path.join("images", "predict images"))
    args = parser.parse_args()

    run_pipeline(args.model_path, args.image_dir)
