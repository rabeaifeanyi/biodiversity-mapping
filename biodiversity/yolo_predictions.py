import os
import glob
import logging
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import networkx as nx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def run_yolo_prediction(
    model_path,
    image_dir,
    max_img_size=1000,
    tile_size=640,
    overlap=100,
    conf_thresh=0.3,
    skip_existing=True
):
    """
    Runs YOLO prediction on images in image_dir.

    Args:
        model_path (str): Path to YOLO model.
        image_dir (str): Directory with images.
        max_img_size (int): Threshold to decide if tiling is used.
        tile_size (int): Size of each tile.
        overlap (int): Overlap between tiles.
        conf_thresh (float): Confidence threshold.
        skip_existing (bool): If True, skip images with existing XMLs.
    """
    logging.info("Loading YOLO model...")
    model = YOLO(model_path)
    model.to(device)

    exts = ['png', 'jpg', 'jpeg', 'JPG', 'JPEG']
    img_files = [f for ext in exts for f in glob.glob(os.path.join(image_dir, f"*.{ext}"))]

    if not img_files:
        logging.warning(f"No images in {image_dir}")
        return

    logging.info(f"{len(img_files)} images will be processed...")

    with torch.no_grad():
        for img_file in img_files:
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            img_path = os.path.join(image_dir, img_file)
            xml_path = os.path.join(image_dir, f"{img_name}.xml")

            if skip_existing and os.path.exists(xml_path):
                logging.info(f"Skipping {img_name}: XML already exists.")
                continue

            logging.info(f"Inference for {img_name}...")

            img = Image.open(img_path)
            w, h = img.size

            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'folder').text = os.path.basename(image_dir)
            ET.SubElement(annotation, 'filename').text = os.path.basename(img_file)
            ET.SubElement(annotation, 'path').text = img_path

            size_elem = ET.SubElement(annotation, 'size')
            ET.SubElement(size_elem, 'width').text = str(w)
            ET.SubElement(size_elem, 'height').text = str(h)
            ET.SubElement(size_elem, 'depth').text = "3"

            ET.SubElement(annotation, 'segmented').text = "0"

            detections = 0

            if max(w, h) > max_img_size:
                logging.info(f"Image {img_name} is large ({w}x{h}), using tiling...")
                for y in range(0, h, tile_size - overlap):
                    for x in range(0, w, tile_size - overlap):
                        x_end = min(x + tile_size, w)
                        y_end = min(y + tile_size, h)

                        tile = img.crop((x, y, x_end, y_end))
                        tile_np = np.array(tile)

                        results = model(tile_np, conf=conf_thresh, imgsz=tile_size, nms=True)

                        for result in results:
                            for box in result.boxes:
                                cls_id = int(box.cls.item())
                                cls_name = result.names[cls_id]
                                conf = box.conf.item()
                                xmin_tile, ymin_tile, xmax_tile, ymax_tile = box.xyxy[0].tolist()

                                xmin = int(xmin_tile + x)
                                xmax = int(xmax_tile + x)
                                ymin = int(ymin_tile + y)
                                ymax = int(ymax_tile + y)

                                detections += 1
                                logging.info(f"Tile [{x}:{x_end}, {y}:{y_end}] detected {cls_name} at [{xmin},{ymin},{xmax},{ymax}]")

                                obj = ET.SubElement(annotation, 'object')
                                ET.SubElement(obj, 'name').text = cls_name
                                ET.SubElement(obj, 'confidence').text = str(round(conf, 2))
                                ET.SubElement(obj, 'pose').text = "Unspecified"
                                ET.SubElement(obj, 'truncated').text = "0"
                                ET.SubElement(obj, 'difficult').text = "0"

                                bbox = ET.SubElement(obj, 'bndbox')
                                ET.SubElement(bbox, 'xmin').text = str(xmin)
                                ET.SubElement(bbox, 'ymin').text = str(ymin)
                                ET.SubElement(bbox, 'xmax').text = str(xmax)
                                ET.SubElement(bbox, 'ymax').text = str(ymax)

            else:
                logging.info(f"Image {img_name} is small enough ({w}x{h}), processing as whole.")
                results = model(img_path, conf=conf_thresh, imgsz=tile_size, nms=True)

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        cls_name = result.names[cls_id]
                        conf = box.conf.item()
                        xmin, ymin, xmax, ymax = [int(x) for x in box.xyxy[0].tolist()]

                        detections += 1
                        logging.info(f"{cls_name} detected at [{xmin},{ymin},{xmax},{ymax}]")

                        obj = ET.SubElement(annotation, 'object')
                        ET.SubElement(obj, 'name').text = cls_name
                        ET.SubElement(obj, 'confidence').text = str(round(conf, 2))
                        ET.SubElement(obj, 'pose').text = "Unspecified"
                        ET.SubElement(obj, 'truncated').text = "0"
                        ET.SubElement(obj, 'difficult').text = "0"

                        bbox = ET.SubElement(obj, 'bndbox')
                        ET.SubElement(bbox, 'xmin').text = str(xmin)
                        ET.SubElement(bbox, 'ymin').text = str(ymin)
                        ET.SubElement(bbox, 'xmax').text = str(xmax)
                        ET.SubElement(bbox, 'ymax').text = str(ymax)

            tree = ET.ElementTree(annotation)
            tree.write(xml_path)
            logging.info(f"XML saved: {xml_path} (total detections: {detections})")




def intersection_over_union(boxA, boxB):
    # This code is copied!
    # Source: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def cluster_boxes(boxes, iou_threshold=0.3):
    """
    boxes: list of dicts {"box":(xmin,ymin,xmax,ymax), "class":str, "conf":float}
    Returns list of clusters (list of lists of indices)
    """
    G = nx.Graph() 
    G.add_nodes_from(range(len(boxes)))
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if boxes[i]["class"] != boxes[j]["class"]:
                continue
            if intersection_over_union(boxes[i]["box"], boxes[j]["box"]) > iou_threshold:
                G.add_edge(i, j)
    clusters = list(nx.connected_components(G))
    return clusters



def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        conf = float(obj.find("confidence").text)
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        objects.append({
            "elem": obj,
            "class": cls,
            "conf": conf,
            "box": (xmin, ymin, xmax, ymax)
        })
    return tree, root, objects



def deduplicate_xml_in_folder(folder, iou_threshold=0.001):
    xml_files = [f for f in os.listdir(folder) if f.lower().endswith(".xml")]
    if not xml_files:
        print("No XML files found.")
        return

    for xml_file in xml_files:
        xml_path = os.path.join(folder, xml_file)
        tree, root, objects = parse_xml(xml_path)

        if len(objects) <= 1:
            continue

        clusters = cluster_boxes(objects, iou_threshold)

        keep_ids = set()
        for cluster in clusters:
            if len(cluster) == 1:
                keep_ids.update(cluster)
            else:
                best = max(cluster, key=lambda i: objects[i]["conf"])
                keep_ids.add(best)

        if len(keep_ids) == len(objects):
            continue

        print(f"[{xml_file}] Removed {len(objects) - len(keep_ids)} duplicates.")

        for obj in root.findall("object"):
            root.remove(obj)
        for i in sorted(keep_ids):
            root.append(objects[i]["elem"])

        tree.write(xml_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate bounding boxes in XML files.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing XML files.")
    args = parser.parse_args()

    deduplicate_xml_in_folder(args.folder)