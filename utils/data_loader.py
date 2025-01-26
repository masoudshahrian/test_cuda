import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, path, img_size=(128, 128)):
        self.path = path
        self.img_size = img_size
        self.image_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # تبدیل به تانسور و نرمال‌سازی به [0, 1]- normalization
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to نرمال‌سازی به [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        #seprate Image into two halves- upper and lower halves تقسیم تصویر به نیمه‌های بالا و پایین
        h, w, _ = img.shape
        upper_half = img[:h // 2, :]
        lower_half = img[h // 2:, :]

        #set the parts-  اعمال تبدیل‌ها
        upper_half = self.transform(upper_half)
        lower_half = self.transform(lower_half)

        return upper_half, lower_half

