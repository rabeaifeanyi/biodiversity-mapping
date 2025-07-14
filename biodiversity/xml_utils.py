import xml.etree.ElementTree as ET
import logging

def xml_unpack(path):
    """
    Parses a PASCAL VOC XML file and extracts detection information.

    Args:
        path (str): Path to the XML annotation file.

    Returns:
        tuple:
            - objects (list of dict): Each detection with:
                - 'name' (str): Class label of the object.
                - 'confidence' (float): Confidence score.
                - 'bbox' (dict): Bounding box coordinates (xmin, ymin, xmax, ymax).
            - width (int): Image width in pixels.
            - height (int): Image height in pixels.

    Logs:
        INFO message indicating how many objects were found.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        xml.etree.ElementTree.ParseError: If the XML file is malformed.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    objects = []
    for obj in root.findall('object'):
        name = obj.findtext('name')
        confidence = float(obj.findtext('confidence'))
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))

        objects.append({
            'name': name,
            'confidence': confidence,
            'bbox': {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }
        })

    logging.info(f"XML geladen: {path}, {len(objects)} Objekte gefunden.")
    return objects, width, height
