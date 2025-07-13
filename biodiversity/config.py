from dataclasses import dataclass

@dataclass
class CameraConfig:
    sensor_width_mm: float = 22.3
    sensor_height_mm: float = 14.9
    focal_length_mm: float = 11
    camera_height_m: float = 0.60

@dataclass
class DroneCameraConfig:
    sensor_width_mm: float = 22.3
    sensor_height_mm: float = 14.9
    focal_length_mm: float = 11
    camera_height_m: float = 9
