import xml.etree.ElementTree as ET

def xml_unpack(pfad):

    # XML-Datei laden mit Pfad-Variable
    tree = ET.parse(pfad)
    root = tree.getroot()

    # Allgemeine Daten auslesen
    folder = root.findtext('folder')
    filename = root.findtext('filename')
    path = root.findtext('path')

    size = root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    depth = int(size.findtext('depth'))

    segmented = int(root.findtext('segmented'))

    # Alle Objekte auslesen (kann mehrere geben)
    objects = []
    for obj in root.findall('object'):
        name = obj.findtext('name')
        confidence = float(obj.findtext('confidence'))
        pose = obj.findtext('pose')
        truncated = int(obj.findtext('truncated'))
        difficult = int(obj.findtext('difficult'))

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))

        obj_dict = {
            'name': name,
            'confidence': confidence,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
            'bbox': {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }
        }
        objects.append(obj_dict)

    # Ausgabe
    print(f"Folder: {folder}")
    print(f"Filename: {filename}")
    print(f"Bildgröße: {width}x{height}x{depth}")
    print(f"Segmentiert: {segmented}")
    print(f"Anzahl Objekte: {len(objects)}")

    for i, obj in enumerate(objects, start=1):
        print(f"\nObjekt {i}:")
        print(f"  Name: {obj['name']}")
        print(f"  Confidence: {obj['confidence']}")
        print(f"  Bounding Box: {obj['bbox']}")

    return objects