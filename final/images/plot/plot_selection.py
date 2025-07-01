import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# TODO maybe provide full path
folder_path = 'D:/Uni/MARS/Project 2.2/Project/images/plot/selection'

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.JPG", "*.JPEG", "*.avif"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
image_paths.sort()

if not image_paths:
    print(f"Keine Bilder im Ordner '{folder_path}' gefunden.")
else:
    num_images = len(image_paths)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax, img_path in zip(axes, image_paths):

        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path))
        ax.axis("off")

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(folder_path, base_name + ".xml")
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                cls = obj.find("name").text
                xmin = int(obj.find("bndbox/xmin").text)
                ymin = int(obj.find("bndbox/ymin").text)
                xmax = int(obj.find("bndbox/xmax").text)
                ymax = int(obj.find("bndbox/ymax").text)

                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(
                    xmin,
                    ymin - 5,
                    cls,
                    color="yellow",
                    fontsize=10,
                    backgroundcolor="red"
                )

    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("ipad_img_test_plot.png")
    plt.show()
