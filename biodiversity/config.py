from dataclasses import dataclass

@dataclass
class CameraConfig:
    sensor_width_mm: float = 22.3
    sensor_height_mm: float = 14.9
    focal_length_mm: float = 11
    camera_height_m: float = 0.60

@dataclass
class DroneCameraConfig:
    sensor_width_mm = 35.9
    sensor_height_mm = 24.0
    focal_length_mm: float = 35.0
    camera_height_m: float = 6.0
