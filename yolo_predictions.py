# This code is nearly identical to a script provided by Deepak HB, slightly modified to fit the project

    # Original file:
    # Copyright 2024 ATB, Potsdam

    # author: Deepak HB
    # file_name: predictions_yolo.py
    # file_description: Python file to get predictions from YOLO models
    # used_with:
    # file_version: 1.0  on 20240927
    # date_created: 20240927
    # guide:
    # remarks:
    # ====================================================================================================================

import os
import glob
import logging
import xml.etree.ElementTree as ET
from PIL import Image # type: ignore
import torch # type: ignore
import torch.backends.cudnn as cudnn # type: ignore
from ultralytics import YOLO # type: ignore

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def run_yolo_prediction(model_path, image_dir):
    logging.info("Loding YOLO model...")
    
    # load model
    model = YOLO(model_path)
    model.to(device)

    exts = ['png', 'jpg', 'jpeg', 'JPG', 'JPEG']
    img_files = [f for ext in exts for f in glob.glob(os.path.join(image_dir, f"*.{ext}"))]

    if not img_files:
        logging.warning(f"No images in {image_dir}")
        return

    logging.info(f"{len(img_files)} images are being processed...")

    with torch.no_grad():
        # read image
        for img_file in img_files:
            
            # annotation file
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            logging.info(f"Inferenz für {img_name}...")

            results = model(img_file, conf=0.3, imgsz=640, nms=True)

            annotation = ET.Element('annotation')
            
            ET.SubElement(annotation, 'folder').text = os.path.basename(image_dir)
            ET.SubElement(annotation, 'filename').text = os.path.basename(img_file)
            ET.SubElement(annotation, 'path').text = img_file

            size_elem = ET.SubElement(annotation, 'size')
            
            with Image.open(img_file) as im:
                width, height = im.size
            
            ET.SubElement(size_elem, 'width').text = str(width)
            ET.SubElement(size_elem, 'height').text = str(height)
            ET.SubElement(size_elem, 'depth').text = "3"

            ET.SubElement(annotation, 'segmented').text = "0"

            for result in results:
                # to save all the inference results
                # result.save_txt(os.path.join(img_dir, 'output.txt'))
                    
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls.item())
                    cls_name = result.names[cls_id]
                    conf = box.conf.item()
                    xmin, ymin, xmax, ymax = [int(x) for x in box.xyxy[0].tolist()]

                    logging.info(f"{cls_name} erkannt mit Konfidenz {conf:.2f} bei [{xmin},{ymin},{xmax},{ymax}]")

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

            xml_path = os.path.join(image_dir, f"{img_name}.xml")
            tree = ET.ElementTree(annotation)
            tree.write(xml_path)
            logging.info(f"XML gespeichert: {xml_path}")
