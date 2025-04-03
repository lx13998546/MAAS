'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-11-05 10:27:34
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-11-21 16:36:04
FilePath: /code/deep-learning/pytorch_classification/Test10_regnet/batch_predict.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json
import torch
import argparse
from PIL import Image, ImageOps
from torchvision import transforms
from model.model import shufflenet_v2_x1_0
import time

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tensor_to_pil(tensor):
    """Converts a tensor to PIL Image."""
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = transforms.ToPILImage()(tensor)
    return tensor

# Resize and pad image to target size (keeping aspect ratio)
def resize_and_pad(image_path, target_size=(224, 224)):
    # Open the image
    img = Image.open(image_path)
    
    # Get original image size
    width, height = img.size
    
    # Calculate scaling ratio to ensure the longer side fits the target size
    if width > height:
        new_width = target_size[0]
        new_height = int(new_width * height / width)
    else:
        new_height = target_size[1]
        new_width = int(new_height * width / height)
    
    # Resize the image
    img_resized = img.resize((new_width, new_height))
    
    # Calculate padding (to center the image)
    left = (target_size[0] - new_width) // 2
    top = (target_size[1] - new_height) // 2
    right = target_size[0] - new_width - left
    bottom = target_size[1] - new_height - top
    
    # Add padding (white padding)
    img_padded = ImageOps.expand(img_resized, (left, top, right, bottom), (255, 255, 255))
    
    return img_padded

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Set root folder path
    assert os.path.exists(args.root_folder), f"Folder '{args.root_folder}' does not exist"

    # Load class labels
    assert os.path.exists(args.json_path), f"File '{args.json_path}' does not exist."
    
    with open(args.json_path, "r") as f:
        class_indict = json.load(f)

    # Create the model
    model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    
    # Load model weights
    assert os.path.exists(args.weight_path), f"Weight file '{args.weight_path}' does not exist."
    model.load_state_dict(torch.load(args.weight_path, map_location=device))

    model.eval()

    start_tim = time.time()

    # Iterate through folders in the root folder
    for folder_name in os.listdir(args.root_folder):
        folder_path = os.path.join(args.root_folder, folder_name)

        # Ensure it's a directory
        if not os.path.isdir(folder_path):
            continue
        
        # MP image folder
        img_folder = os.path.join(folder_path, "PIC")
        if not os.path.exists(img_folder):
            continue
        
        # Create an output folder for predicted images
        output_folder = os.path.join(folder_path, 'predicted_images')
        create_directory(output_folder)

        # Iterate through image files in the folder
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_name)

            # Ensure it's an image file
            if not img_name.lower().endswith(('png', 'jpg', 'tif')):
                continue
            
            original_img = resize_and_pad(img_path)
            img_format = original_img.format  # Get the original image format

            # Apply the transformations for prediction
            img_tensor = data_transform(original_img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)  # Add batch dimension

            with torch.no_grad():
                # Predict the class
                output = torch.squeeze(model(img_tensor.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            # Get predicted class and probability
            predicted_class = class_indict[str(predict_cla)]
            
            # Create a directory for the predicted class if it doesn't exist
            class_folder = os.path.join(output_folder, predicted_class)
            create_directory(class_folder)

            # Save the original image (not the transformed one) in the predicted class folder
            save_path = os.path.join(class_folder, img_name)
            original_img.save(save_path, format=img_format)  # Save using the original format

            print(f"Saved original image {img_name} to {class_folder}")
            
    end_tim = time.time()
    print(f"Total time taken: {end_tim - start_tim:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True, help='Root directory for dataset')
    parser.add_argument('--json_path', type=str, required=True, help='Path to class indices JSON file')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to pretrained weights')
    
    args = parser.parse_args()
    main(args)
