from PIL import Image, ImageOps 
import os

def resize_and_pad(image_path, target_size=(224, 224)):
    
    img = Image.open(image_path)
    
    width, height = img.size
    
    # Calculate scaling ratio
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
    
    # Fill in blank areas
    img_padded = ImageOps.expand(img_resized, (left, top, right, bottom), (255, 255, 255))
    
    return img_padded

def process_images_in_folder(folder_path, target_size=(224, 224), save_folder="resized_images"):

    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            output_subfolder = os.path.join(save_folder, subfolder_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                
                if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                    img = resize_and_pad(file_path, target_size)
                    
                    file_extension = file_name.split('.')[-1].lower()
                    save_path = os.path.join(output_subfolder, f"resized_{file_name}")
                    
                    if file_extension in ['jpg', 'jpeg']:
                        img.save(save_path, format='JPEG')
                    elif file_extension == 'png':
                        img.save(save_path, format='PNG')
                    else:
                        img.save(save_path)
                    
                    print(f"Processed: {file_name}, saved to {save_path}")
                    
folder_path = '/data/original_images/'  # The folder path of the original image
output_folder = '/data/resized_images/'  # New save path
process_images_in_folder(folder_path, target_size=(224, 224), save_folder=output_folder)





