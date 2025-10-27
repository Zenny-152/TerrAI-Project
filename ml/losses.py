import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        alpha: None | scalar | list/tuple | numpy.ndarray | torch.Tensor
        If per-class weights are desired, pass a sequence with length == num_classes.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        else:
            # normalize alpha into a 1D torch tensor (not nested)
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.clone().float()
            elif isinstance(alpha, (list, tuple, np.ndarray)):
                self.alpha = torch.tensor(list(alpha), dtype=torch.float)
            else:
                # scalar value
                self.alpha = torch.tensor([alpha], dtype=torch.float)

    def forward(self, inputs, targets):
        # inputs: logits [B, C], targets: [B] (long)
        logp = F.log_softmax(inputs, dim=1)               # [B,C]
        p = torch.exp(logp)                               # [B,C]
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (p * targets_onehot).sum(dim=1)              # [B] prob for true class
        log_pt = (logp * targets_onehot).sum(dim=1)       # [B]
        if self.alpha is not None:
            # alpha can be per-class or scalar
            if self.alpha.numel() == 1:
                at = self.alpha.to(inputs.device)
            else:
                # per-class: index per target
                at = self.alpha.to(inputs.device)[targets]
        else:
            at = 1.0
        loss = - at * ((1 - pt) ** self.gamma) * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
