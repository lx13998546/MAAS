import os
import sys
import json
import torch
import argparse
import logging
from PIL import Image
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model import shufflenet_v2_x1_0

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def plot_confusion_matrix(y_true, y_pred, classes, save_path, font_size=14, dpi=300):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={'size': font_size})
    plt.xlabel('True', fontsize=font_size)
    plt.ylabel('Predicted', fontsize=font_size)
    plt.title('Confusion Matrix', fontsize=font_size)
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    create_directory(args.weights_dir)
    
    data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    image_path = os.path.abspath(args.data_root)
    assert os.path.exists(image_path), f"{image_path} path does not exist."
    
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info(f"Using {len(train_loader.dataset)} images for training, {len(val_loader.dataset)} images for validation.")
    
    model = shufflenet_v2_x1_0(num_classes=args.num_classes)
    model.to(device)
    
    assert os.path.exists(args.weight_path), f"Weight file {args.weight_path} does not exist."
    pre_weights = torch.load(args.weight_path, map_location='cpu')
    pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(pre_dict, strict=False)
    
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)    
    best_acc = 0.0
    log_file_path = args.save_dir
    log_file = open(log_file_path, 'w')
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)
        
        model.eval()
        all_preds = []
        all_labels = []
        acc = 0.0
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.numpy())
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, args.epochs)

        val_accurate = acc / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)
        log_info = '[epoch %d] train_loss: %.3f  val_loss: %.3f  val_accuracy: %.3f\n' % (
            epoch + 1, running_loss / len(train_loader), avg_val_loss, val_accurate)
        print(log_info)
        log_file.write(log_info)
        
        class_names = list(val_dataset.classes)
        plot_confusion_matrix(all_labels, all_preds, classes=class_names,
                              save_path=os.path.join(args.save_dir, 'confusion_matrix.png'))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), args.save_dir)

    print('Finished Training')
    log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to pretrained weights')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save weights and logs')
    
    args = parser.parse_args()
    main(args)

