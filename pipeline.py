"""
Biodiversity Mapping Pipeline

This script implements a complete pipeline to:
- Run YOLO object detection on input images.
- Parse the resulting XML annotations.
- Transform detected bounding boxes into world coordinates.
- Cache the coordinates to avoid recomputation.
- Generate a scatter plot of all detected plant positions.

Usage:
    python pipeline.py --model_path ... --image_dir ...
"""

import argparse
import os
from pathlib import Path

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
    Returns:
        tuple:
            - fig (matplotlib.figure.Figure or None)
            - logs (list of str)
            - stats (dict)
            - global_pos (list of detections)
    """
    logs = []
    def log(msg):
        logs.append(msg)

    log("Starting pipeline...")
    output_dir = Path(image_dir)
    cache_path = output_dir / "coordinates_cache.pkl"

    cache = load_cache(cache_path)
    log("Cache loaded.")

    if cache and any(k != "_stats" for k in cache.keys()):
        log("Using existing cache, skipping YOLO prediction.")
        global_pos = []
        for k, coords in cache.items():
            if k == "_stats":
                continue
            global_pos.extend(coords)

        if not global_pos:
            log("Cache was empty, no coordinates to plot.")
            return None, logs, {}, []

        stats = cache.get("_stats", {})
        counts = list(stats.get("detections_per_image", {}).values())
        distribution = {}
        for c in counts:
            distribution.setdefault(c, 0)
            distribution[c] += 1
        stats["detections_per_image_distribution"] = distribution

        fig = plot_map(global_pos)
        return fig, logs, stats, global_pos

    run_yolo_prediction(model_path, image_dir)
    log("YOLO prediction completed.")

    camera_cfg = CameraConfig()
    global_pos = []
    stats = {
        "detections_per_image": {},
        "detections_per_class": {}
    }

    for file in os.listdir(output_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            basename = os.path.splitext(file)[0]
            xml_path = output_dir / f"{basename}.xml"

            if not xml_path.exists():
                log(f"{basename}: XML missing, skipping.")
                continue

            objects, width, height = xml_unpack(xml_path)
            coords = []

            log(f"Processing {file}: {len(objects)} detections found.")
            stats["detections_per_image"][file] = len(objects)

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
                    coords.append((result[0], result[1], obj["name"], obj["confidence"]))
                    global_pos.append((result[0], result[1], obj["name"], obj["confidence"]))
                    stats["detections_per_class"].setdefault(obj["name"], 0)
                    stats["detections_per_class"][obj["name"]] += 1

            cache[file] = coords

    cache["_stats"] = stats
    save_cache(cache_path, cache)
    log(f"Saved cache with {len(global_pos)} total coordinates.")

    log("Summary of detections per image:")
    for img, count in stats["detections_per_image"].items():
        log(f" - {img}: {count} detections")
    log("Summary of detections per class:")
    for cls, count in stats["detections_per_class"].items():
        log(f" - {cls}: {count} detections")

    if not global_pos:
        log("No detections.")
        return None, logs, stats, []

    fig = plot_map(global_pos)
    return fig, logs, stats, global_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biodiversity Mapping Pipeline")
    parser.add_argument("--model_path", type=str, default=os.path.join("model", "best.pt"))
    parser.add_argument("--image_dir", type=str, default=os.path.join("images", "predict images"))
    args = parser.parse_args()

    run_pipeline(args.model_path, args.image_dir)
