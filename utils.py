# utils.py

import torch

def custom_collate(batch):
    """
    Custom collate function for handling variable-length patch sequences in DataLoader.

    Args:
        batch (list): List of samples from Dataset.
                      Each item is a tuple: (a_patches, b_patches, label, case_id)
                        - a_patches: Tensor [Nᵢ, 3, 224, 224]
                        - b_patches: Tensor [Nᵢ, 3, 224, 224]
                        - label: int
                        - case_id: str

    Returns:
        tuple: (list of a_tensors, list of b_tensors, labels tensor, list of case_ids)
    """
    a_list, b_list, label_list, cid_list = zip(*batch)

    labels = torch.tensor(label_list, dtype=torch.long)
    return list(a_list), list(b_list), labels, list(cid_list)
