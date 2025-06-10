import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms


class ImageFolderWithoutLabels(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        A custom dataset for images stored in a single directory without subfolders or labels.

        Args:
        - root_dir (str): Path to the directory containing images.
        - transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List all files in the directory
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if self.is_image_file(f)
        ]
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in directory: {root_dir}")

    @staticmethod
    def is_image_file(filename):
        """Check if a file is an image based on its extension."""
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
        return any(filename.lower().endswith(ext) for ext in extensions)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return an image at the given index.

        Args:
        - idx (int): Index of the image to load.

        Returns:
        - image (Tensor): The transformed image.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB image
        if self.transform:
            image = self.transform(image)
        return image
