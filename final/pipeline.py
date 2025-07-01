# this file makes up the pipeline and has the goal to run all the steps at once to realise the heat map
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import argparse

from coordinates import main_coordinates
from read_xml import xml_unpack


#-----------------------------------
# Argumente parsen
#-----------------------------------
parser = argparse.ArgumentParser(description="Biodiversity mapping pipeline")
parser.add_argument("--model_path", type=str, default=os.path.join("model", "best.pt"),
                    help="Path to YOLO model file")
parser.add_argument("--image_dir", type=str, default=os.path.join("images", "predict images"),
                    help="Directory with input images")
parser.add_argument("--output_dir", type=str, default=os.path.join("images", "predict images", "all"),
                    help="Directory with detection results and JSONs")
args = parser.parse_args()

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# YOLO Prediction zuerst aufrufen
def run_yolo_prediction():
    prediction_script = os.path.join('predict', 'predictions_yolo.py')
    print("🔁 Starte YOLO-Vorhersage...")

    result = subprocess.run([
        'python',
        prediction_script,
        '-s', args.model_path,
        '-i', args.image_dir
    ], text=True)  
    #], capture_output=True, text=True) 

    if result.returncode == 0:
        print("✅ YOLO-Vorhersage abgeschlossen.")
    else:
        print("❌ Fehler bei der YOLO-Vorhersage:")
        print(result.stderr)
        exit(1)

run_yolo_prediction()


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

#verzeichnis
#dateien_ordner = os.path.join('images', 'predict images','all')

# Kameraparameter
sensor_width_mm = 22.3
sensor_height_mm = 14.9
focal_length_mm = 11
camera_height_m = 0.60

# Ergebnisliste mit absoluten UTM zone n33 Koordinaten
global_pos = []

# Durchlaufe alle Bilddateien
for datei in os.listdir(args.output_dir):
    if datei.lower().endswith(('.png', '.jpg', '.jpeg')):
        basisname = os.path.splitext(datei)[0]
        xml_pfad = os.path.join(args.output_dir, basisname + '.xml')
        json_pfad = os.path.join(args.output_dir, basisname + '.json')

        if os.path.exists(xml_pfad) and os.path.exists(json_pfad):
            pflanzen = xml_unpack(xml_pfad)

            for plant in pflanzen:
                plant_pos_x = (plant['bbox']['xmin'] + plant['bbox']['xmax']) / 2
                plant_pos_y = (plant['bbox']['ymin'] + plant['bbox']['ymax']) / 2

                global_xy = main_coordinates(
                    plant_pos_x,
                    plant_pos_y,
                    sensor_width_mm,
                    sensor_height_mm,
                    focal_length_mm,
                    camera_height_m,
                    args.output_dir,
                    datei
                )

                global_pos.append(global_xy)
        else:
            print(f"Überspringe {basisname}: XML oder JSON fehlt.")

print(f"\n✅ Insgesamt {len(global_pos)} Pflanzenpositionen berechnet.")


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

#TODO heat map aus Koordinaten erstellen


if global_pos:
    x_coords = np.array([p[0] for p in global_pos])
    y_coords = np.array([p[1] for p in global_pos])

    xy = np.vstack([x_coords, y_coords])
    kde = gaussian_kde(xy)

    x_grid, y_grid = np.mgrid[
        x_coords.min():x_coords.max():100j,
        y_coords.min():y_coords.max():100j
    ]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords).reshape(x_grid.shape)

    plt.figure(figsize=(10, 8))
    plt.title("🌿 Pflanzen Heatmap (UTM Zone 33N)")
    plt.xlabel("UTM X (m)")
    plt.ylabel("UTM Y (m)")
    plt.imshow(z.T, origin='lower',
               extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
               cmap='hot', alpha=0.7)
    plt.colorbar(label='Relative Dichte')
    plt.scatter(x_coords, y_coords, s=10, c='blue', label='Pflanzenpositionen', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("❌ Keine Koordinaten vorhanden -> keine Heatmap erstellt.")
