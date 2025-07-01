
# Biodiversity Mapping Project 2025

In provided code change `YOLOv10` to `YOLO`.

Command for running the script:

```bash
python predictions_yolo.py \                                  
-s /home/rabea/Documents/MWRS/Indicators_Mapping/train/weights/best.pt \ #TODO add your path
-i /home/rabea/Documents/MWRS/biodiversity-mapping/ipad_img1 
```

My environment is in `mwrs.yaml`. I created the environment like this:

```bash
micromamba create --name mwrs python=3.10
micromamba activate mwrs
micromamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  
micromamba install numpy pillow   
pip install ultralytics      
```

## Measurements

Camera heights:

1. 0.545 m (up until 3 pm)
2. 0.67 m

Camera specs:
6.34 Brennweite (10 mm)

## How to run the pipeline

```bash
 % python pipeline.py \
  --model_path [...Indicators_Mapping/train/weights/best.pt] \
  --image_dir [path to your data] \
  --output_dir [path to your data/results]
```