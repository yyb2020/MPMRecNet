# loss/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks to address class imbalance.
    
    Args:
        alpha (Tensor or None): Class weights tensor, e.g., [0.25, 0.75]. If None, no weighting is applied.
        gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        reduction (str): 'mean', 'sum', or 'none' to determine how to reduce the loss across the batch.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute focal loss.

        Args:
            logits (Tensor): Predicted logits, shape (B, num_classes).
            targets (Tensor): Ground truth labels, shape (B,).

        Returns:
            Tensor: Scalar loss if reduction='mean' or 'sum', otherwise (B,) loss vector.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
