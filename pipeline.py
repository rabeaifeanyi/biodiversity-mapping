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
    python pipeline.py --model_path ... --image_dir ... --output_dir ...
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

from yolo_predictions import run_yolo_prediction
from xml_utils import xml_unpack
from geo_utils import main_coordinates
from cache_utils import load_cache, save_cache
from config import CameraConfig

def run_pipeline(model_path, image_dir, output_dir):
    """
    Runs the full detection and mapping pipeline.

    Args:
        model_path (str): Path to the YOLO .pt weights file.
        image_dir (str): Directory containing input images.
        output_dir (str): Directory where outputs (XML, cache, etc.) are saved.

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
    output_dir = Path(output_dir)
    cache_path = output_dir / "coordinates_cache.pkl"

    cache = load_cache(cache_path)
    log("Cache loaded.")

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

                    if result:
                        coords.append(result)
                        global_pos.append(result)

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

    fig = plt.figure(figsize=(10, 8))
    plt.title("🌿 Plant Heatmap (UTM Zone 33N)")
    plt.xlabel("UTM X (m)")
    plt.ylabel("UTM Y (m)")
    plt.imshow(z.T, origin='lower',
               extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
               cmap='hot', alpha=0.7)
    plt.colorbar(label='Relative Density')
    plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.4)
    plt.tight_layout()

    log("Heatmap generated successfully.")
    return fig, logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biodiversity Mapping Pipeline")
    parser.add_argument("--model_path", type=str, default=os.path.join("model", "best.pt"))
    parser.add_argument("--image_dir", type=str, default=os.path.join("images", "predict images"))
    parser.add_argument("--output_dir", type=str, default=os.path.join("images", "predict images", "all"))
    args = parser.parse_args()

    run_pipeline(args.model_path, args.image_dir, args.output_dir)
