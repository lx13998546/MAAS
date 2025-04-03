import cv2
import os
import random
import shutil
import numpy as np

def load_labels(label_path):
    """Load Label File"""
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            labels.append((cls_id, bbox))
    return labels

def save_labels(output_label_path, labels):
    """Save Label File"""
    with open(output_label_path, 'w') as f_out:
        for label in labels:
            cls_id, bbox = label
            f_out.write(f"{cls_id} {' '.join(map(str, bbox))}\n")

def random_contrast_adjustment(img):
    """Random Contrast Adjustment"""
    alpha = random.uniform(0.2, 1.0)  # alpha: The larger the value, the stronger the contrast
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def random_flip(img, labels):
    """Random Horizontal and Vertical Flips"""
    flipped_labels = labels.copy()

    if random.random() < 0.5:  # 50% probability horizontal flip
        img = cv2.flip(img, 1)
        # Adjust the value of x_center
        for i in range(len(flipped_labels)):
            cls_id, bbox = flipped_labels[i]
            x_center, y_center, width, height = bbox
            flipped_labels[i] = (cls_id, [1 - x_center, y_center, width, height])

    if random.random() < 0.5:  # 50% probability vertical flip
        img = cv2.flip(img, 0)
        # Adjust the value of y_center
        for i in range(len(flipped_labels)):
            cls_id, bbox = flipped_labels[i]
            x_center, y_center, width, height = bbox
            flipped_labels[i] = (cls_id, [x_center, 1 - y_center, width, height])

    return img, flipped_labels



def augment_images_with_samples(img_path, label_path, sample_img_paths, output_img_folder, output_label_folder, sample_type, num_augments=1):
    """Enhance Original Images"""
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    labels = load_labels(label_path)

    for i in range(num_augments):
        augmented_img = img.copy()
        new_labels = labels.copy()

        # Random enhancement input images
        augmented_img = random_contrast_adjustment(augmented_img)
        augmented_img = random_flip(augmented_img, new_labels) 

        for sample_img_path in sample_img_paths:
            sample_img = cv2.imread(sample_img_path)
            sample_img, _ = random_flip(sample_img, []) 

            sample_h, sample_w, _ = sample_img.shape

            attempt = 0
            while attempt < 10:  # Try to find suitable non overlapping areas
                cut_x = random.randint(0, w - sample_w)
                cut_y = random.randint(0, h - sample_h)

                cut_region = (cut_x, cut_y, cut_x + sample_w, cut_y + sample_h)

                overlap = False
                for label in new_labels:
                    _, (x_center, y_center, box_w, box_h) = label
                    box_x1 = int((x_center - box_w / 2) * w)
                    box_y1 = int((y_center - box_h / 2) * h)
                    box_x2 = int((x_center + box_w / 2) * w)
                    box_y2 = int((y_center + box_h / 2) * h)

                    if not (cut_region[2] < box_x1 or cut_region[0] > box_x2 or cut_region[3] < box_y1 or cut_region[1] > box_y2):
                        overlap = True
                        break

                if not overlap:
                    augmented_img[cut_y:cut_y + sample_h, cut_x:cut_x + sample_w] = sample_img

                    if sample_type == 'positive': 
                        new_x_center = (cut_x + sample_w / 2) / w
                        new_y_center = (cut_y + sample_h / 2) / h
                        new_w = sample_w / w
                        new_h = sample_h / h
                        new_labels.append((0, [new_x_center, new_y_center, new_w, new_h])) 

                    break
                attempt += 1

        # Save enhanced images and labels
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_img_path = os.path.join(output_img_folder, f"{base_name}_aug_{sample_type}_{i+1}.tif")
        output_label_path = os.path.join(output_label_folder, f"{base_name}_aug_{sample_type}_{i+1}.txt")

        cv2.imwrite(output_img_path, augmented_img)
        save_labels(output_label_path, new_labels)


def process_images(input_img_folder, input_label_folder, pos_sample_folder, neg_sample_folder, output_img_folder, output_label_folder, num_augments=2):
    """Process All Images"""
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    img_files = [f for f in os.listdir(input_img_folder) if f.endswith(('.jpg', '.tif'))]
    pos_sample_files = [f for f in os.listdir(pos_sample_folder) if f.endswith(('.jpg', '.tif'))]
    neg_sample_files = [f for f in os.listdir(neg_sample_folder) if f.endswith(('.jpg', '.tif'))]

    for img_file in img_files:
        img_path = os.path.join(input_img_folder, img_file)
        label_path = os.path.join(input_label_folder, img_file.replace('.jpg', '.txt').replace('.tif', '.txt'))

        if not os.path.exists(label_path):
            continue

        # 1. Save original images and labels
        shutil.copy(img_path, os.path.join(output_img_folder, img_file))
        shutil.copy(label_path, os.path.join(output_label_folder, img_file.replace('.jpg', '.txt').replace('.tif', '.txt')))

        # 2. Generate enhanced data with positive samples
        pos_sample_paths = random.sample(pos_sample_files, random.randint(2, 3))
        pos_sample_paths = [os.path.join(pos_sample_folder, f) for f in pos_sample_paths]
        augment_images_with_samples(img_path, label_path, pos_sample_paths, output_img_folder, output_label_folder, 'positive', num_augments)

        # 3. Generate enhanced data with negative samples
        neg_sample_paths = random.sample(neg_sample_files, random.randint(2, 3))
        neg_sample_paths = [os.path.join(neg_sample_folder, f) for f in neg_sample_paths]
        augment_images_with_samples(img_path, label_path, neg_sample_paths, output_img_folder, output_label_folder, 'negative', num_augments)

input_img_folder = '/data/original_images/'
input_label_folder = '/data/original_labels/'
pos_sample_folder = '/data/Positive_data/'
neg_sample_folder = '/data/Negative_data/'
output_img_folder = '/data/Augemented_images/'
output_label_folder = '/data/Augemented_labels/'

process_images(input_img_folder, input_label_folder, pos_sample_folder, neg_sample_folder, output_img_folder, output_label_folder, num_augments=1)



