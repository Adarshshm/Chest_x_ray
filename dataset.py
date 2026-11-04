import os
import numpy as np
from PIL import Image
from imblearn.over_sampling import BorderlineSMOTE
import torch
from torch.utils.data import Dataset, DataLoader

class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for managing chest X-ray tensors and labels.
    
    >>> import numpy as np
    >>> dummy_images = np.zeros((2, 128, 128, 3), dtype=np.float32)
    >>> dummy_labels = np.array([0, 1])
    >>> dataset = ChestXRayDataset(dummy_images, dummy_labels)
    >>> len(dataset)
    2
    """
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_split_dir(split_dir, img_size=(128, 128)):
    """
    Loads images and labels from a specific directory split (train, val, or test).
    Performs image resizing to 128x128 and normalizes pixel values to the [0, 1] range.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(split_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names) if os.path.isdir(os.path.join(split_dir, cls_name))}

    for cls_name, idx in class_to_idx.items():
        cls_path = os.path.join(split_dir, cls_name)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                images.append(img_arr)
                labels.append(idx)
            except Exception:
                continue
                
    return np.array(images), np.array(labels), class_names

def get_dataloaders(base_dir="./dataset", batch_size=32):
    """
    Loads train, validation, and test folders independently to prevent data leakage.
    Applies Borderline-SMOTE exclusively to the training dataset to handle class imbalance.
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    X_train, y_train, class_names = load_split_dir(train_dir)
    X_val, y_val, _ = load_split_dir(val_dir)
    X_test, y_test, _ = load_split_dir(test_dir)

    n_samples, h, w, c = X_train.shape
    X_train_flat = X_train.reshape(n_samples, -1)
    
    bl_smote = BorderlineSMOTE(random_state=42)
    X_train_resampled_flat, y_train_resampled = bl_smote.fit_resample(X_train_flat, y_train)
    X_train_resampled = X_train_resampled_flat.reshape(-1, h, w, c)

    train_loader = DataLoader(ChestXRayDataset(X_train_resampled, y_train_resampled), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ChestXRayDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ChestXRayDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(class_names), class_names