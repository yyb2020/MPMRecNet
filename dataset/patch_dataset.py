# dataset/patch_dataset.py

import os
import torch
import random
from torch.utils.data import Dataset

class MultiModalPatchDataset(Dataset):
    """
    Custom dataset for loading multi-modal .pt patch files for each case.
    Each .pt file should contain a dictionary with keys mapping to sub-patch tensors:
        {
            "0": {"a": tensor(H, W), "b": tensor(H, W)},
            ...
        }
    These patches will be expanded to 3 channels and stacked.
    """

    def __init__(self, pt_dir: str, case_ids: list, labels: list, max_patches: int = None):
        """
        Args:
            pt_dir (str): Path to directory with .pt patch files.
            case_ids (List[str]): List of case IDs (without .pt extension).
            labels (List[int]): Corresponding label for each case.
            max_patches (int): Optional limit on the number of patches per case.
        """
        self.pt_dir = pt_dir
        self.case_ids = case_ids
        self.labels = labels
        self.max_patches = max_patches

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        label = self.labels[idx]

        pt_path = os.path.join(self.pt_dir, f"{case_id}.pt")
        patch_dict = torch.load(pt_path)

        def to_3channel(tensor_1ch):
            return tensor_1ch.repeat(3, 1, 1)  # Expand single channel to 3 channels

        patch_keys = list(patch_dict.keys())

        if self.max_patches is not None and len(patch_keys) > self.max_patches:
            patch_keys = random.sample(patch_keys, self.max_patches)

        a_patches = torch.stack([to_3channel(patch_dict[k]["a"]) for k in patch_keys])
        b_patches = torch.stack([to_3channel(patch_dict[k]["b"]) for k in patch_keys])

        return a_patches, b_patches, torch.tensor(label).long(), case_id
