# Biodiversity Mapping Pipeline

This repository provides a pipeline for detecting plants in field images, mapping their pixel positions to real-world coordinates, and generating spatial visualizations (e.g., heatmaps). The core script is `pipeline.py`, supported by several helper modules. The pipeline can be used for either robot or drone imagery.

## Prerequisites

- Python 3.8+ environment (eg. micromamba or conda)
  ```bash
   micromamba create --name mwrs python=3.10
   micromamba activate mwrs
   micromamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  
   micromamba install numpy pillow   
   pip install ultralytics      
  ```
- A trained YOLO model file (e.g., `best.pt`)
- A folder of input images (PNG/JPG/JPEG)
- JSON metadata files alongside images, containing camera pose

## Repository Structure

```
├── model/
│   └── best.pt               # YOLO model weights
├── images/
│   └── predict images/       # Input images
│       └── *.png, *.jpg      # Photos to process
├── pipeline.py               # Main pipeline script
├── cache_utils.py            # Load/save prediction cache
├── yolo_predictions.py       # Run YOLO inference → XML annotations
├── xml_utils.py              # Parse XML detection results
├── geo_utils.py              # Pixel→world conversion functions
├── plot_map.py               # Plotting funcs
└── config.py                 # Camera parameters (sensor, lens, height)
```

## Usage

You have two options to run the pipeline:

1. **Terminal version**  
   Run directly in your console:
   ```bash
   python pipeline.py --model_path /path/to/best.pt --image_dir /path/to/images
   ```

2. **Streamlit Web Interface**  
   If you prefer a browser interface where you can enter parameters and see logs and plots interactively, install Streamlit:

   ```bash
   pip install streamlit
   ```

   Then start the app:

   ```bash
   streamlit run app.py
   ```

   In the web interface, you will see fields to enter:
   - `model_path` (the YOLO weights file)
   - `image_dir` (the folder with input images)

   The pipeline does exactly the same processing steps (YOLO prediction, coordinate mapping, caching, and plotting) as the terminal version, but shows all logs and the heatmap in your browser.

## Functionality

1. **YOLO Inference** (`yolo_predictions.py`)
   - Loads model via `ultralytics.YOLO`.
   - Runs detection on each image (confidence threshold 0.3).
   - Writes PASCAL-VOC XML files with `<object>` entries.

2. **Cache Management** (`cache_utils.py`)
   - `load_cache()`: Loads or initializes a `pickle` cache (`coordinates_cache.pkl`).
   - `save_cache()`: Stores computed coordinates to avoid re-processing.

3. **XML Parsing** (`xml_utils.py`)
   - `xml_unpack()`: Reads XML, returns list of detections (`xmin`,`ymin`,`xmax`,`ymax`) and image size.

4. **Pixel → World Conversion** (`geo_utils.py`)
   - **JSON Pose Loading**: `json_unpack()` reads camera translation `(tx, ty, tz)` and quaternion rotation `(rx, ry, rz, rw)`.
   - **Flattening**: `flatten_json()` flattens nested JSON keys.
   - **Pixel to Ground Distances** (`pixel_to_world`):

     ```
     sensor_width = sensor_width_mm / 1000
     pixel_size_x = sensor_width / width_px
     dx = (x_px - width_px/2) * pixel_size_x
     x_trans = dx * (camera_height_m / focal_length)
     ```

     (analogous for `y`)

   - **Transform to World Frame** (`transform`):

     ```
     x_real = cos(r_z) * x_trans - sin(r_z) * y_trans
     y_real = sin(r_z) * x_trans + cos(r_z) * y_trans
     X_world = t_x + x_real
     Y_world = t_y + y_real
     ```

     where `r_z` is the rotation around the vertical axis in degrees.

5. **Plot Generation** (`pipeline.py`)
   - Simple scatter plot of coordinates

## Outputs

1. **XML files**: Detection annotations per image.
2. **Cache**: `coordinates_cache.pkl` storing `{ image_filename: [(X,Y),…] }`.
3. **Heatmap**: A Matplotlib window (or saved figure) showing plant-density.
4. **Logs**: INFO/WARNING messages about progress, missing files, etc.

## Installation

Make sure you have **Python 3.8 or newer** installed.

1. **Clone this repository:**
   ```bash
   git clone https://github.com/rabeaifeanyi/biodiversity-mapping.git
   cd biodiversity-pipeline
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional – install Streamlit** if you want to use the web interface:
   ```bash
   pip install streamlit
   ```

### Streamlit Interface

To launch the browser-based UI:

```bash
streamlit run app.py
```

## Configuration

Edit `config.py` to set your camera parameters:

```python
class CameraConfig:
    sensor_width_mm = 6.4
    sensor_height_mm = 4.8
    focal_length_mm = 3.6
    camera_height_m = 0.5
```

Adjust for your hardware.

## TODOs

- [x] Add requirements
- [x] Integration of drone image processing
- [x] Improvement of caching
- [x] Better results plot
- [x] Underlay of satelite images (temproary fix)
- [ ] Underlay of real stiched picture
- [ ] Heatmap
- [ ] Drone camera config and drone flag in terminal
- [ ] Examples
