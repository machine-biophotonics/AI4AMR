from typing import Optional

import torch


def compute_iou_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> float:
    """Compute the Intersection over Union metric for the predictions and labels.

    Args:
        y_hat (torch.Tensor): The prediction of dimensions (B, C, H, W), C being
            equal to the number of classes.
        y (torch.Tensor): The label for the prediction of dimensions (B, H, W)
        ignore_index (int | None, optional): ignore label to omit predictions in
            given region.
        eps (float, optional): To smooth the division and prevent division
        by zero. Defaults to 1e-6.

    Returns:
        float: The mean IoU
    """
    num_classes = int(y.max().item() + 1)
    y_hat = torch.argmax(y_hat, dim=1)
    
    ious = []
    for c in range(num_classes):
        y_hat_c = (y_hat == c)
        y_c = (y == c)

        # Ignore all regions with ignore
        if ignore_index is not None:
            mask = (y != ignore_index)
            y_hat_c = y_hat_c & mask
            y_c = y_c & mask

        intersection = (y_hat_c & y_c).sum().float()
        union = (y_hat_c | y_c).sum().float()

        if union > 0:
            ious.append((intersection + eps) / (union + eps))

    return torch.mean(torch.stack(ious))
