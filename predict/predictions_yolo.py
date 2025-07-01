# Copyright 2024 ATB, Potsdam
#
# author: Deepak HB
# file_name: predictions_yolo.py
# file_description: Python file to get predictions from YOLO models
# used_with:
# file_version: 1.0  on 20240927
# date_created: 20240927
# guide:
# remarks:
# ====================================================================================================================

import argparse
import glob
import os
import numpy as np
import xml.etree.ElementTree as ET

import warnings

from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO


warnings.filterwarnings('ignore')  # suppress warnings

# verify whether your system has GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# enable benchmark to speed up the inference
cudnn.benchmark = True

# predictions from the model
def model_predictions(saved_model, img_dir):
    """
    :param saved_model: saved model (model.pt)
    :param img: images for predictions
    :param labels_dir: directory to h
    :return: prediction_results
    """

    # load model
    model = YOLO(saved_model)
    model.to(device)

    exts = ['png', 'jpg', 'jpeg', 'gif', 'JPG', 'JPEG']
    img_files = [img_file for ext in exts for img_file in glob.glob(img_dir + '/*.' + ext)]

    # # the input needs to be a tensor
    # preprocess = transforms.Compose([
    #     # convert the frame to a CHW torch tensor for training
    #     transforms.ToTensor(),
    # ])
    

    with torch.no_grad():
        while True:
            # read image
            for img_file in img_files:

                # annotation file
                img_path_name, img_file_name = os.path.split(img_file)
                img_name, extension = img_file_name.split('.')

                annotation = ET.Element('annotation')

                ET.SubElement(annotation, 'folder').text = str('unknown')
                ET.SubElement(annotation, 'filename').text = str(img_file_name)
                ET.SubElement(annotation, 'path').text = str('unknown')

                source = ET.SubElement(annotation, 'source')
                ET.SubElement(source, 'database').text = 'unknown'

                # read image
                read_img = Image.open(img_file)  # read_image
                img_width, img_height = read_img.size

                size = ET.SubElement(annotation, 'size')
                ET.SubElement(size, 'width').text = str(img_width)
                ET.SubElement(size, 'height').text = str(img_height)
                ET.SubElement(size, 'depth').text = str(3)

                ET.SubElement(annotation, 'segmented').text = str(0)

                # running inference
                print('Running inference for {}...'.format(img_file, end=''))


                # The model can handle multiple images simultaneously so we need to add an
                # empty dimension for the batch.
                # [C, H, W] -> [1, C, H, W]
                # img_tensor = preprocess(img_np).unsqueeze(0)
                # run the model
                results = model(img_file, conf=0.3, imgsz=640, nms=True)

                for result in results:
                    # to save all the inference results
                    # result.save_txt(os.path.join(img_dir, 'output.txt'))
                    a = 0
                    for i in range(len(result.boxes)):
                        cls_name = result.names[int(result.boxes.cls[a].item())]
                        bbox = result.boxes.xyxy[a].tolist()
                        score = result.boxes.conf[a].item()
                        a += 1
                        print('Results for {} and bbox {}: cls_name: {}, bboxes: {}, score:{}'.
                              format(img_file, a, cls_name, bbox, score))

                        object_element = ET.SubElement(annotation, 'object')
                        ET.SubElement(object_element, 'name').text = str(cls_name)
                        ET.SubElement(object_element, 'confidence').text = str(round(score, 2))
                        ET.SubElement(object_element, 'pose').text = str('Unspecified')
                        ET.SubElement(object_element, 'truncated').text = str(0)
                        ET.SubElement(object_element, 'difficult').text = str(0)

                        bnd_box = ET.SubElement(object_element, 'bndbox')
                        ET.SubElement(bnd_box, 'xmin').text = str(int(bbox[0]))
                        ET.SubElement(bnd_box, 'ymin').text = str(int(bbox[1]))
                        ET.SubElement(bnd_box, 'xmax').text = str(int(bbox[2]))
                        ET.SubElement(bnd_box, 'ymax').text = str(int(bbox[3]))

                tree = ET.ElementTree(annotation)
                tree.write(os.path.join(img_dir, img_name + '.xml'))

            return None

import os

def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description='A file to infer the results from the deep learning model and'
                    ' create autonomous annotation files.')

    parser.add_argument('-s',
                        '--savedModel',
                        help='path to the saved model, trained from similar objects trying to infer.',
                        type=str)

    parser.add_argument('-i',
                        '--imgDir',
                        help='path to image files ',
                        type=str, default=None)

    args = parser.parse_args()


    try:
        # List all items in the directory
        saved_pred_model = args.savedModel
        path = args.imgDir

        for item in os.listdir(path):
            img_dir = os.path.join(path, item)
            model_predictions(saved_pred_model, img_dir)

            # Check if the item is a directory
            if os.path.isdir(img_dir):
                print(f"Directory: {dir}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()