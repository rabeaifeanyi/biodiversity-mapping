import json
import numpy as np #type:ignore
import os
import logging
from PIL import Image
import PIL.ExifTags as ExifTags
from pyproj import Transformer

def json_unpack(folder, image_file):
    """
    This function loads a JSON file associated with a given image file.

    Args:
        folder (str): The directory containing the JSON file.
        image_file (str): The name of the image file (e.g., 'image01.jpg').

    Returns:
        dict or None: The JSON data as a dictionary, or None if the file is missing.
    """
    basename = os.path.splitext(image_file)[0]
    json_path = os.path.join(folder, f"{basename}.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        logging.info(f"JSON loaded: {json_path}")
        return data
    
    else:
        logging.warning(f"No JSON found: {json_path}")
        return None

def flatten_json(data, prefix=''):
    """
    Recursively flattens a JSON dictionary into a flat dictionary
    with keys joined by underscores.

    Args:
        data (dict): The dictionary to flatten.
        prefix (str, optional): A prefix for the keys during recursion.

    Returns:
        dict: The flattened dictionary.
    """
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_json(value, full_key))
        else:
            flat[full_key] = value
    return flat

def pixel_to_world(x_pixel, y_pixel, width, height, sensor_width_mm, sensor_height_mm, focal_length_mm, camera_height_m):
    """
    Converts a pixel position in the image to distances relative to the camera center.

    Args:
        x_pixel (float): X coordinate in pixels.
        y_pixel (float): Y coordinate in pixels.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        sensor_width_mm (float): Camera sensor width in millimeters.
        sensor_height_mm (float): Camera sensor height in millimeters.
        focal_length_mm (float): Camera focal length in millimeters.
        camera_height_m (float): Camera height above the ground in meters.

    Returns:
        tuple: (x_trans, y_trans) - distances in meters relative to the camera origin.
    """
    sensor_width = sensor_width_mm / 1000
    sensor_height = sensor_height_mm / 1000
    focal_length = focal_length_mm / 1000
    cx = width / 2
    cy = height / 2
    pixel_size_x = sensor_width / width
    pixel_size_y = sensor_height / height

    dx_sensor = (x_pixel - cx) * pixel_size_x
    dy_sensor = (y_pixel - cy) * pixel_size_y

    x_trans = dx_sensor * (camera_height_m / focal_length)
    y_trans = dy_sensor * (camera_height_m / focal_length)

    return x_trans, y_trans

def transform(transform_translation_x, transform_translation_y, transform_rotation_z, x_trans, y_trans):
    """
    Applies a rotation and translation to local coordinates to obtain absolute world coordinates.

    Args:
        transform_translation_x (float): X translation offset in meters.
        transform_translation_y (float): Y translation offset in meters.
        transform_rotation_z (float): Rotation angle in degrees.
        x_trans (float): X coordinate in meters relative to the camera.
        y_trans (float): Y coordinate in meters relative to the camera.

    Returns:
        tuple: (absolute_x, absolute_y) - absolute coordinates in the reference frame.
    """
    x_real = np.cos(np.deg2rad(transform_rotation_z)) * x_trans - np.sin(np.deg2rad(transform_rotation_z)) * y_trans
    y_real = np.sin(np.deg2rad(transform_rotation_z)) * x_trans + np.cos(np.deg2rad(transform_rotation_z)) * y_trans

    absolute_x = transform_translation_x + x_real
    absolute_y = transform_translation_y + y_real

    return float(absolute_x), float(absolute_y)

def extract_gps_from_exif(image_path):
    """
    Extracts GPS Latitude and Longitude from EXIF.
    Returns (lat, lon) or None.
    """
    img = Image.open(image_path)
    exif_data = img._getexif()

    if not exif_data:
        return None

    gps_info = None
    for tag, value in exif_data.items():
        decoded = ExifTags.TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_info = value
            break

    if not gps_info:
        return None

    def _convert(coord, ref):
        d = float(coord[0])
        m = float(coord[1])
        s = float(coord[2])
        sign = -1 if ref in ["S", "W"] else 1
        return sign * (d + m/60 + s/3600)


    lat = _convert(gps_info[2], gps_info[1])
    lon = _convert(gps_info[4], gps_info[3])

    return (lat, lon)

def get_utm_transformer(lat, lon):
        """
        Gibt einen Transformer von WGS84 → UTM passend zur Position zurück.
        
        Args:
            lat (float): Breitengrad
            lon (float): Längengrad
            
        Returns:
            Transformer: pyproj Transformer-Objekt
        """
        # Berechne UTM-Zone
        zone = int((lon + 180) / 6) + 1
        # Bestimme Nord/Süd
        hemisphere = 'north' if lat >= 0 else 'south'
        
        # Baue UTM CRS-String
        if hemisphere == 'north':
            epsg_code = 32600 + zone  # Nordhalbkugel
        else:
            epsg_code = 32700 + zone  # Südhalbkugel
            
        print(f"UTM Zone: {zone} {hemisphere} (EPSG:{epsg_code})")
        
        # Erzeuge Transformer
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
        return transformer

def main_coordinates(x_pixel, y_pixel, camera_cfg, folder, image_file, width, height):
    """
    Computes the absolute world coordinates for a pixel position in an image.

    This function:
    - Loads camera metadata from a JSON file.
    - Converts the pixel position to ground-relative distances.
    - Applies camera rotation and translation to get global coordinates.

    Args:
        x_pixel (float): X pixel coordinate of the target.
        y_pixel (float): Y pixel coordinate of the target.
        camera_cfg (CameraConfig): Camera configuration object with sensor and lens parameters.
        folder (str): Directory containing the JSON metadata.
        image_file (str): Image filename.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        tuple or None: (absolute_x, absolute_y) coordinates in meters, or None if JSON is missing.
    """
    data = json_unpack(folder, image_file)
    if data:
        flat = flatten_json(data)
        tx = flat["transform_translation_x"]
        ty = flat["transform_translation_y"]
        rz = flat["transform_rotation_z"]

        x_trans, y_trans = pixel_to_world(
            x_pixel, y_pixel,
            width, height,
            camera_cfg.sensor_width_mm,
            camera_cfg.sensor_height_mm,
            camera_cfg.focal_length_mm,
            camera_cfg.camera_height_m
        )

        absolute_x, absolute_y = transform(tx, ty, rz, x_trans, y_trans)
        return absolute_x, absolute_y

    else: # drone case
        gps = extract_gps_from_exif(os.path.join(folder, image_file))
        if not gps:
            return None

        lat, lon = gps

        altitude = camera_cfg.camera_height_m

        x_trans, y_trans = pixel_to_world(
            x_pixel, y_pixel,
            width, height,
            camera_cfg.sensor_width_mm,
            camera_cfg.sensor_height_mm,
            camera_cfg.focal_length_mm,
            altitude
        )

    transformer = get_utm_transformer(lat, lon)
    tx, ty = transformer.transform(lon, lat)


    absolute_x = tx + x_trans
    absolute_y = ty + y_trans

    return absolute_x, absolute_y

