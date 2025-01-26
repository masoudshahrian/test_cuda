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
            transforms.ToTensor(),  # تبدیل به تانسور و نرمال‌سازی به [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # نرمال‌سازی به [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        # تقسیم تصویر به نیمه‌های بالا و پایین
        h, w, _ = img.shape
        upper_half = img[:h // 2, :]
        lower_half = img[h // 2:, :]

        # اعمال تبدیل‌ها
        upper_half = self.transform(upper_half)
        lower_half = self.transform(lower_half)

        return upper_half, lower_half


# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
#
#
# class HalfImageDataset(Dataset):
#     def __init__(self, root_dir, img_size=(128, 256)):
#         """
#         پارامترها:
#             root_dir (str): مسیر فولدر حاوی تصاویر
#             img_size (tuple): (height, width) برای تغییر اندازه تصاویر
#         """
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
#
#         # تبدیل‌های پیش‌پردازش
#         self.transform = transforms.Compose([
#             transforms.Resize(img_size),
#             transforms.ToTensor(),
#         ])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_path).convert('RGB')
#
#         # اعمال تبدیلات
#         image = self.transform(image)
#
#         # تقسیم تصویر به نیمه بالا و پایین
#         _, H, W = image.shape
#         upper_half = image[:, :H // 2, :]  # (C, H//2, W)
#         lower_half = image[:, H // 2:, :]  # (C, H//2, W)
#
#         return upper_half, lower_half
#
#
# # -----------------------------------------------------------
# # مثال استفاده از دیتالودر
# # -----------------------------------------------------------
# if __name__ == "__main__":
#     # پارامترها
#     DATA_DIR = "path/to/your/images"
#     BATCH_SIZE = 16
#     IMG_SIZE = (64, 128)  # (H, W) نیمه بالا و پایین بعد از تقسیم
#
#     # ایجاد دیتاست و دیتالودر
#     dataset = HalfImageDataset(DATA_DIR, img_size=(2 * IMG_SIZE[0], IMG_SIZE[1]))
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     # تست یک بچ
#     for upper, lower in dataloader:
#         print(f"نیمه بالایی شکل: {upper.shape}")  # باید (B, 3, 64, 128) باشد
#         print(f"نیمه پایینی شکل: {lower.shape}")  # باید (B, 3, 64, 128) باشد
#         break