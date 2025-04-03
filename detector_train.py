import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to 0, 1, etc., depending on which GPU you want to use
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')
# Model configuration file
model_yaml_path = r"./ultralytics/cfg/models/11/malaria_detctor.yaml"
# Dataset configuration file
data_yaml_path = r"./data/malaria.yaml"

if __name__ == '__main__':

    model = YOLO(model_yaml_path)
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=250,
                          batch=12,
                          workers=0,
                          optimizer='SGD',  # using SGD
                          amp=False, 
                          project='./runs/',
                          name='train1',
    )
