# this file uses the geodata (json file) and converts the pixel data of the plant into global coordinates

import json
import os
import numpy as np



def json_unpack(ordner, bild_datei):

    # 🔧 Basisname extrahieren (ohne Endung)
    basename = os.path.splitext(bild_datei)[0]

    # 🔍 Passendes JSON suchen
    json_path = os.path.join(ordner, f'{basename}.json')

    # ✅ JSON-Datei einlesen, falls vorhanden
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ JSON geladen: {json_path}")
    
    else:
        print(f"❌ Kein passendes JSON gefunden für {bild_datei}")
        return None

    return data


# json file verarbeiten
def flatten_json(data, prefix=''):
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_json(value, full_key))
        else:
            flat[full_key] = value
    return flat



# transforms data from function pixel_to_world into UTM zone33 coordinates
def transform(transform_translation_x, transform_translation_y, transform_rotation_z, x_trans, y_trans): 

    x_real = x_trans + np.cos(np.deg2rad(transform_rotation_z)) * x_trans
    y_real = y_trans + np.sin(np.deg2rad(transform_rotation_z)) * y_trans

    absolute_x = transform_translation_x + x_real
    absolute_y = transform_translation_y + y_real

    return float(absolute_x), float(absolute_y), float(x_real), float(y_real)


# transforms pixel position into projected distance (x,y) from image center/camera position
def pixel_to_world(x_pixel, y_pixel,                        
                   width, height,
                   sensor_width_mm, sensor_height_mm,
                   focal_length_mm,
                   camera_height_m):

    # Sensorgröße in m
    sensor_width = sensor_width_mm / 1000
    sensor_height = sensor_height_mm / 1000
    focal_length = focal_length_mm / 1000

    # Bildmitte in Pixeln
    cx = width / 2
    cy = height / 2

    # Pixelgrößen in m/Pixel
    pixel_size_x = sensor_width / width
    pixel_size_y = sensor_height / height

    # Abstand vom Zentrum in Metern auf dem Sensor
    dx_sensor = (x_pixel - cx) * pixel_size_x
    dy_sensor = (y_pixel - cy) * pixel_size_y

    # Ähnlichkeitsbeziehung (Strahlensatz): x_real / dx_sensor = z / f
    x_trans = dx_sensor * (camera_height_m / focal_length)
    y_trans = dy_sensor * (camera_height_m / focal_length)

    return x_trans, y_trans


# main steps for the transformation
def main_coordinates(x_pixel, y_pixel,sensor_width_mm,sensor_height_mm,focal_length_mm,camera_height_m, ordner, bild_datei):

    data = json_unpack(ordner, bild_datei)
    if data:
        flat_data = flatten_json(data)
    else:
        return None, None

    # In Variablen schreiben (vorsichtig mit globals)
    for key, value in flat_data.items():
        globals()[key] = value

    # transform pixels to distance
    x_trans, y_trans = pixel_to_world(
    x_pixel, y_pixel,
    width, height,
    sensor_width_mm, sensor_height_mm,
    focal_length_mm,
    camera_height_m)

    absolute_x, absolute_y, x_real, y_real = transform(transform_translation_x, transform_translation_y, transform_rotation_z, x_trans, y_trans)

    return absolute_x, absolute_y
