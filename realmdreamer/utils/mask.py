import cv2
import numpy as np
import torch


def get_outpainting_mask(mask: torch.Tensor):  # B 1 H W
    """
    Filter the mask to only retain the parts connected to the image border
    """

    mask = mask.squeeze(1)

    mask_labels = torch.zeros_like(mask.float())
    for i in range(mask.shape[0]):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask[0].detach().cpu().numpy().astype(np.uint8)
        )
        mask_labels[i] = torch.from_numpy(labels)
    mask_labels = mask_labels.to(mask.device)

    # Generate pixel coordinates for border of image
    b, h, w = mask.shape
    x = torch.arange(w).repeat(h, 1).to(mask.device)
    y = torch.arange(h).repeat(w, 1).transpose(0, 1).to(mask.device)

    # Get the border mask - border of size 1
    border_size = 1
    border_mask = (
        (x < border_size)
        | (x > w - border_size)
        | (y < border_size)
        | (y > h - border_size)
    )

    # Repeat the border mask for each batch
    border_mask = border_mask.repeat(b, 1, 1)

    # Get the indices where mask[border_mask] = 1
    mask_border = mask.clone()
    mask_border[~border_mask] = 0  # Set all non-border pixels to 0

    border_indices_valid = torch.nonzero(
        mask_border, as_tuple=False
    )  # Get all indices where the mask is 1

    test_mask = torch.zeros_like(mask)
    test_mask[
        border_indices_valid[:, 0],
        border_indices_valid[:, 1],
        border_indices_valid[:, 2],
    ] = 1

    # Get the labels of the border pixels
    border_labels = mask_labels[
        border_indices_valid[:, 0],
        border_indices_valid[:, 1],
        border_indices_valid[:, 2],
    ]

    # Only keep the mask where the label correspond to a label in border_labels
    mask_labels_all = torch.zeros_like(mask_labels).bool()

    for label in torch.unique(border_labels):
        mask_labels_all = mask_labels_all | (mask_labels == label)
        mask_class = mask_labels == label

    mask = (
        mask.bool() & mask_labels_all
    )  # Only keep the mask where the label corresponding to something touching the border

    return mask.float().unsqueeze(1)
