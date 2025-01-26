import sys
import os

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
from utils.data_loader import CelebADataset
from models.autoencoder import Autoencoder

# تنظیمات
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_path = "D:/celebA/celeba-dataset/img_align_celeba/test"
model_path = "models/autoencoder.pth"

# بارگذاری داده‌ها
test_dataset = CelebADataset(test_path, img_size=(128, 128))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

# بارگذاری مدل
model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# نمایش نتایج
with torch.no_grad():
    for upper_half, lower_half in test_loader:
        upper_half, lower_half = upper_half.to(device), lower_half.to(device)
        outputs = model(upper_half)

        # نمایش تصاویر
        fig, axes = plt.subplots(3, 10, figsize=(15, 5))
        for i in range(10):
            axes[0, i].imshow(upper_half[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)  # نرمال‌سازی معکوس
            axes[1, i].imshow(outputs[i].cpu().permute(1, 2, 0))
            axes[2, i].imshow(torch.cat([upper_half[i].cpu(), outputs[i].cpu()], dim=1).permute(1, 2, 0))

        plt.show()
        break

