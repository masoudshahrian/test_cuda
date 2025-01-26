import sys
import os

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.data_loader import CelebADataset
from models.autoencoder import Autoencoder

# تنظیمات
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = "D:/celebA/celeba-dataset/img_align_celeba/train"
batch_size = 16
epochs = 2
learning_rate = 0.001

# بارگذاری داده‌ها
train_dataset = CelebADataset(train_path, img_size=(128, 128))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# مدل، تابع خطا و بهینه‌ساز
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# آموزش مدل
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for upper_half, lower_half in train_loader:
        upper_half, lower_half = upper_half.to(device), lower_half.to(device)

        optimizer.zero_grad()
        outputs = model(upper_half)
        loss = criterion(outputs, lower_half)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# ذخیره مدل
torch.save(model.state_dict(), "models/autoencoder.pth")



