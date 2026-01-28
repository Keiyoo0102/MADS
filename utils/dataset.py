import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class SandDuneDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=False, text_embed_path=None):
        """
        Args:
            root_dir (str): Path to dataset (containing img_dir, ann_dir).
            split (str): 'train', 'val', or 'test'.
            transform (bool): Apply data augmentation (flips/rotations).
            text_embed_path (str, optional): Path to .npy text embeddings. If provided, returns text feats.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.text_embeds = None

        # Paths
        self.img_dir = os.path.join(root_dir, 'img_dir', split)
        self.ann_dir = os.path.join(root_dir, 'ann_dir', split)

        # Load Text Embeddings if provided
        if text_embed_path:
            if os.path.exists(text_embed_path):
                text_np = np.load(text_embed_path)
                self.text_embeds = torch.from_numpy(text_np).float()
            else:
                print(f"Warning: Text embedding path provided but not found: {text_embed_path}")

        # Get file list
        self.files = []
        glob_pattern = os.path.join(self.img_dir, "*.npy")
        file_paths = glob.glob(glob_pattern)
        for fp in file_paths:
            filename = os.path.basename(fp).replace('.npy', '')
            # Check label existence
            ann_path = os.path.join(self.ann_dir, filename + '.png')
            if os.path.exists(ann_path):
                self.files.append(filename)

        print(f"[{split}] Loaded {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]

        # 1. Load Image (64-channel Embedding)
        img_path = os.path.join(self.img_dir, filename + '.npy')
        image = np.load(img_path) # (64, 512, 512)

        # 2. Load Label
        ann_path = os.path.join(self.ann_dir, filename + '.png')
        label = np.array(Image.open(ann_path))

        # 3. Augmentation
        if self.transform:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()

        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).long()

        if self.text_embeds is not None:
            return image_tensor, self.text_embeds, label_tensor
        else:
            return image_tensor, label_tensor