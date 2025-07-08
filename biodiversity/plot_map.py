import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_map(global_pos, background_path=None):
    """
    Plots plant detections on a map.

    Args:
        global_pos (list of tuples): Each tuple is (x, y, class_name, confidence)
        background_path (str, optional): Path to background image

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12,10))

    # Optional background
    if background_path:
        bg_img = Image.open(background_path)
        ax.imshow(bg_img, extent=[
            min(p[0] for p in global_pos),
            max(p[0] for p in global_pos),
            min(p[1] for p in global_pos),
            max(p[1] for p in global_pos)
        ])
    
    # Unique classes
    classes = sorted(set(p[2] for p in global_pos))
    colors = plt.cm.tab10.colors

    for idx, class_name in enumerate(classes):
        xs = [p[0] for p in global_pos if p[2]==class_name]
        ys = [p[1] for p in global_pos if p[2]==class_name]
        ax.scatter(xs, ys, s=20, label=class_name, color=colors[idx % len(colors)], alpha=0.8)

    ax.set_xlabel("X (UTM m)")
    ax.set_ylabel("Y (UTM m)")
    ax.set_title("Detected Plants Overlay Map")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig
