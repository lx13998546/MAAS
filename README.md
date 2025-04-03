# [Malaria-Assisted Analysis System](XXXXXXXXX)
This project provides a deep learning-based system, which employs ensemble learning techniques to autonomously detect Plasmodium-infected cells and distinguish between the ring and non-ring stages of Plasmodium in thin blood smear images, providing clinical diagnostic support for pathologists.

## Installation
`conda` virtual environment is recommended. 
```
conda create -n MAAS python=3.8.18
conda activate MAAS
pip install -r requirements.txt
```

## Demo
- Integrate the pre-trained detector and classifier to detect and classify all images under the specified folder
```shell script
python demo.py --input_folder test_data_dir
               --detection_output ./data/detector_result/ 
               --classification_output ./data/classifier_result/ 
               --detector_weight ./weight/detector.pt 
               --classifier_weight ./weight/classifier.pth 
               --class_indices ./class_indices.json
               --img_size 640
               --conf_threshold 0.6
```

## Data Preparation
#### Your Dataset
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA should be a directory.
# For eg.:
# YOUR_DATA/
#  detector/
#          images/
#                train/
#                val/
#          labels/
#                train/
#                val/
#  classifier/
#            train/
#                 ClassA/
#                 ClassB/
#                 ClassC/
#                 ......
#            val/
#                 ClassA/
#                 ClassB/
#                 ClassC/
#                 ......
```

## Training

- To train malaria detector on the **detector data**:
```shell script
python detector_train.py
# model_yaml_path: Path to the model yaml file
# data_yaml_path: Path to the data yaml file
# project_name: Item name of test results
```
- To train malaria classifier on the **classifier data**:
```shell script
python train.py --data_root your_data_directory 
                --num_classes Number of classes 
                --weight_path pretrained_weights.pth
                --save_dir save_weights_and_logs_directory
                --save_path /path/to/save/best_model.pth 
```

## Inference

### To inference with the pretrained models on images:
- Detection of Plasmodium Infected Cells with **Detector**:
```shell script
python detector_predict.py
# /DATA/FOLDER/: Folder of the image to be detected
# /detector/output/: Folder of the detection results
# project_name: Item name of test results
```
- Stage classification of Plasmodium Infected Cells with **Classifier**:
```shell script
python classifier_predict.py --root_folder cell_images_dir 
                             --json_path path_class_indices.json
                             --num_classes Number of classes
                             --weight_path ./save_dir/classifier.pth
```

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics).

Thanks for the great implementations! 
