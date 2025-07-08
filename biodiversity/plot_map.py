import matplotlib.pyplot as plt
from PIL import Image

def plot_map(global_pos):
    """
    Plots plant detections on a map.

    Args:
        global_pos (list of tuples): Each tuple is (x, y, class_name, confidence)
        background_path (str, optional): Path to background image

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8,10))
    
    classes = sorted(set(p[2] for p in global_pos))
    colors = plt.cm.tab10.colors

    for idx, class_name in enumerate(classes):
        xs = [p[0] for p in global_pos if p[2]==class_name]
        ys = [p[1] for p in global_pos if p[2]==class_name]
        count = len(xs)
        ax.scatter(
            xs, ys,
            s=20,
            label=f"{class_name} ({count})",
            color=colors[idx % len(colors)],
            alpha=0.8
        )

    ax.set_xlabel("X coordinate (meters)")
    ax.set_ylabel("Y coordinate (meters)")
    ax.set_title("Detected Plants Overlay Map")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig
