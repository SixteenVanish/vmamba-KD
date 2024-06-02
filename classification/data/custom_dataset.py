import os
import torch

from .cached_image_folder import default_img_loader

    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, root_dir, transform=None):
        self.annotations = open(ann_file, "r").readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_img_loader

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        line = self.annotations[idx]
        img_path, label = line.strip().split(" ")  # 拆分每一行以获取图像路径和标签
        img = self.loader(os.path.join(self.root_dir, img_path))
        if self.transform:
            img = self.transform(img)
        label = int(label)  # 将标签转换为整数
        return img, label
    