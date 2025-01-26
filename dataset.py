import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ================== Dataset Class ==================
class ImageDataset(Dataset):
    def __init__(self, path, img_size=(128, 128)):
        self.img_size = img_size
        self.image_paths = glob.glob(os.path.join(path, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        upper_half = img[:, :img.shape[1] // 2, :]
        lower_half = img[:, img.shape[1] // 2:, :]
        return upper_half, lower_half






