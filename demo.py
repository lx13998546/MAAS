import os
import json
import time
import torch
import argparse
from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics import YOLO
from model.model import shufflenet_v2_x1_0

# Create directory if it doesn't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Initialize the YOLO model for object detection
def detect_objects(detector_model, input_folder, output_folder, img_size=640, conf_threshold=0.6):
    detector_model.predict(input_folder, save=True, save_txt=False, save_crop=True,
                           imgsz=img_size, conf=conf_threshold, project=output_folder,
                           name="detected_images")

# Load the classification model (ShuffleNetV2)
def load_classifier_model(device, model_path, class_indices_path):
    model = shufflenet_v2_x1_0(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(class_indices_path, 'r') as f:
        class_indict = json.load(f)

    return model, class_indict

# Resize and pad image to target size (keeping aspect ratio)
def resize_and_pad(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    width, height = img.size

    if width > height:
        new_width = target_size[0]
        new_height = int(new_width * height / width)
    else:
        new_height = target_size[1]
        new_width = int(new_height * width / height)

    img_resized = img.resize((new_width, new_height))

    left = (target_size[0] - new_width) // 2
    top = (target_size[1] - new_height) // 2
    right = target_size[0] - new_width - left
    bottom = target_size[1] - new_height - top

    img_padded = ImageOps.expand(img_resized, (left, top, right, bottom), (255, 255, 255))
    
    return img_padded

# Classify images in the crop folder
def classify_images(classifier_model, class_indict, crop_folder, output_folder, data_transform, device):
    for img_name in os.listdir(crop_folder):
        img_path = os.path.join(crop_folder, img_name)
        if not img_name.lower().endswith(('png', 'jpg', 'tif')):
            continue

        try:
            original_img = resize_and_pad(img_path)
            img_tensor = data_transform(original_img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

            with torch.no_grad():
                output = torch.squeeze(classifier_model(img_tensor)).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            predicted_class = class_indict[str(predict_cla)]
            predicted_prob = predict[predict_cla].numpy()

            class_folder = os.path.join(output_folder, predicted_class)
            create_directory(class_folder)

            save_path = os.path.join(class_folder, img_name)
            original_img.save(save_path, format=original_img.format)

            print(f"Image {img_name} classified as {predicted_class} ({predicted_prob:.4f}) and saved.")
        
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            continue

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    create_directory(args.detection_output)
    create_directory(args.classification_output)

    detector_model = YOLO(args.detector_weight)

    start_time = time.time()
    detect_objects(detector_model, args.input_folder, args.detection_output, args.img_size, args.conf_threshold)
    print(f"Detection completed in {time.time() - start_time:.2f} seconds.")

    classifier_model, class_indict = load_classifier_model(device, args.classifier_weight, args.class_indices)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    crop_folder = os.path.join(args.detection_output, "detected_images", "crops", "PIC")

    start_time = time.time()
    classify_images(classifier_model, class_indict, crop_folder, args.classification_output, data_transform, device)
    print(f"Classification completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object Detection and Classification Pipeline")

    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing input images")
    parser.add_argument('--detection_output', type=str, required=True, help="Output folder for detection results")
    parser.add_argument('--classification_output', type=str, required=True, help="Output folder for classification results")
    parser.add_argument('--detector_weight', type=str, required=True, help="Path to YOLO model weight file")
    parser.add_argument('--classifier_weight', type=str, required=True, help="Path to ShuffleNet classifier weight file")
    parser.add_argument('--class_indices', type=str, required=True, help="Path to class indices JSON file")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for detection model")
    parser.add_argument('--conf_threshold', type=float, default=0.6, help="Confidence threshold for object detection")

    args = parser.parse_args()
    main(args)



