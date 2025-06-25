# this file makes up the pipeline and has the goal to run all the steps at once to realise the heat map

from coordinates import main_coordinates
from read_xml import xml_unpack
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np



#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# YOLO Prediction zuerst aufrufen
def run_yolo_prediction():
    model_path = os.path.join('model', 'best.pt')
    image_dir = os.path.join('images', 'predict images')
    prediction_script = os.path.join('predict', 'predictions_yolo.py')

    print("🔁 Starte YOLO-Vorhersage...")

    result = subprocess.run([
        'python',
        prediction_script,
        '-s', model_path,
        '-i', image_dir
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ YOLO-Vorhersage abgeschlossen.")
    else:
        print("❌ Fehler bei der YOLO-Vorhersage:")
        print(result.stderr)
        exit(1)

# Dann beginnt dein restlicher Code
run_yolo_prediction()



#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

#verzeichnis
dateien_ordner = os.path.join('images', 'predict images','all')

# Kameraparameter
sensor_width_mm = 22.3
sensor_height_mm = 14.9
focal_length_mm = 11
camera_height_m = 0.60

# Ergebnisliste mit absoluten UTM zone n33 Koordinaten
global_pos = []

# Durchlaufe alle Bilddateien
for datei in os.listdir(dateien_ordner):  # Bilder und XMLs liegen beide in images/all
    if datei.lower().endswith(('.png', '.jpg', '.jpeg')):
        basisname = os.path.splitext(datei)[0]

        # Pfade zu zugehörigen Dateien
        xml_pfad = os.path.join(dateien_ordner, basisname + '.xml')
        json_pfad = os.path.join(dateien_ordner, basisname + '.json')  # JSON liegt eine Ebene höher
        bild_pfad = os.path.join(dateien_ordner, datei)

        # Verarbeite nur, wenn XML & JSON existieren
        if os.path.exists(xml_pfad) and os.path.exists(json_pfad):
            pflanzen = xml_unpack(xml_pfad)

            for plant in pflanzen:
                plant_pos_x = (plant['bbox']['xmin'] + plant['bbox']['xmax']) / 2
                plant_pos_y = (plant['bbox']['ymin'] + plant['bbox']['ymax']) / 2

                # Globale Position berechnen
                global_xy = main_coordinates(
                    plant_pos_x,
                    plant_pos_y,
                    sensor_width_mm,
                    sensor_height_mm,
                    focal_length_mm,
                    camera_height_m,
                    dateien_ordner,
                    datei  # Bilddatei übergeben
                )

                global_pos.append(global_xy)
        else:
            print(f"⚠️  Überspringe {basisname}: XML oder JSON fehlt.")

print(f"\n✅ Insgesamt {len(global_pos)} Pflanzenpositionen berechnet.")
print(global_pos)



#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

#TODO heat map aus Koordinaten erstellen


# In separate Arrays zerlegen
x_coords = np.array([p[0] for p in global_pos])
y_coords = np.array([p[1] for p in global_pos])

# Erzeuge 2D-Dichtekarte mit gaussian_kde
xy = np.vstack([x_coords, y_coords])
kde = gaussian_kde(xy)

# Raster zum Plotten erzeugen
x_grid, y_grid = np.mgrid[x_coords.min():x_coords.max():100j, y_coords.min():y_coords.max():100j]
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
z = kde(grid_coords).reshape(x_grid.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.title("🌿 Pflanzen Heatmap (UTM Zone 33N)")
plt.xlabel("UTM X (m)")
plt.ylabel("UTM Y (m)")
plt.imshow(z.T, origin='lower', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], cmap='hot', alpha=0.7)
plt.colorbar(label='Relative Dichte')
plt.scatter(x_coords, y_coords, s=10, c='blue', label='Pflanzenpositionen', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()