import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer

def plot_map(global_pos, r=20, zoom=17):
    """
    Plots:
    - Top: Basemap + Scatter (Übersicht)
    - Bottom: Nah-Detail ohne Basemap
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,14))

    xmin = min(p[0] for p in global_pos) - r
    xmax = max(p[0] for p in global_pos) + r
    ymin = min(p[1] for p in global_pos) - r
    ymax = max(p[1] for p in global_pos) + r

    rx = (xmax - xmin) * 0.2
    ry = (ymax - ymin) * 0.2

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    xmin_detail = x_center - rx
    xmax_detail = x_center + rx
    ymin_detail = y_center - ry
    ymax_detail = y_center + ry

    classes = sorted(set(p[2] for p in global_pos))
    colors = plt.cm.tab10.colors

    for idx, class_name in enumerate(classes):
        xs = [p[0] for p in global_pos if p[2]==class_name]
        ys = [p[1] for p in global_pos if p[2]==class_name]
        count = len(xs)
        ax1.scatter(
            xs, ys,
            s=30,
            label=f"{class_name} ({count})",
            color=colors[idx % len(colors)],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5
        )
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ctx.add_basemap(
        ax1,
        crs="EPSG:32633",
        source=ctx.providers.Esri.WorldImagery,
        zoom=zoom
    )
    ax1.set_xlabel("X coordinate (meters)")
    ax1.set_ylabel("Y coordinate (meters)")
    ax1.set_title("Overview")
    ax1.legend()
    ax1.grid(True)

    for idx, class_name in enumerate(classes):
        xs = [p[0] for p in global_pos if p[2]==class_name]
        ys = [p[1] for p in global_pos if p[2]==class_name]
        count = len(xs)
        ax2.scatter(
            xs, ys,
            s=40,
            label=f"{class_name} ({count})",
            color=colors[idx % len(colors)],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6
        )
    ax2.set_xlim(xmin_detail, xmax_detail)
    ax2.set_ylim(ymin_detail, ymax_detail)
    ax2.set_xlabel("X coordinate (meters)")
    ax2.set_ylabel("Y coordinate (meters)")
    ax2.set_title("Detailed View (Zoomed In)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    t = Transformer.from_crs("EPSG:32633","EPSG:4326")
    lon, lat = t.transform(361088.77, 5815180.05)
    print(f"Sample WGS84: lon {lon:.6f}, lat {lat:.6f}")

    return fig
