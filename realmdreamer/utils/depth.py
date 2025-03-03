import numpy as np
import torch
import cv2
import kornia
import math
import pdb
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef


def batched_index_select_nd(t, inds):
    """
    Performs an index select on dim 1 of an n-dimensional batched tensor.

    :param t: Tensor of shape (batch, n, ...)
    :param inds: Tensor of indices of shape (batch, k)
    :return: Tensor of shape (batch, k, ...)
    """
    return t.gather(
        1, inds[(...,) + (None,) * (len(t.shape) - 2)].expand(-1, -1, *t.shape[2:])
    )


def depth_ranking_loss_multi_patch_masked(
    rendered_depth,
    sampled_depth,
    mask,
    num_patches=16,
    num_pairs=1024,
    margin=1e-4,
    crop_size=(64, 64),
):
    """
    Compute the depth ranking loss between the rendered depth and the sampled depth. The loss is computed by sampling patches from the depth maps and computing the ranking loss between points in the patches.

    The mask provided will be used to mask out the patches where the loss is computed

    :param rendered_depth: The rendered depth map [B 1 H W]
    :param sampled_depth: The sampled depth map [B 1 H W]
    :param mask: The mask to use to mask out the patches [B 1 H W]

    :return depth_loss: The depth ranking loss
    """

    assert (
        rendered_depth.shape == sampled_depth.shape
    ), "Rendered depth and sampled depth must have the same shape"
    assert (
        rendered_depth.shape == mask.shape
    ), "Rendered depth and mask must have the same shape"

    B = rendered_depth.shape[0]

    crop_height, crop_width = crop_size
    top = torch.randint(
        0,
        rendered_depth.shape[-2] - crop_height + 1,
        (num_patches,),
        device=rendered_depth.device,
    )
    left = torch.randint(
        0,
        rendered_depth.shape[-1] - crop_width + 1,
        (num_patches,),
        device=rendered_depth.device,
    )

    # Perform the crop - from a B C H W obain B C num_patches crop_height crop_width
    rows = (
        torch.arange(0, crop_height, device=rendered_depth.device)
        .unsqueeze(1)
        .expand(crop_height, crop_width)
        .reshape(-1)
    )
    cols = (
        torch.arange(0, crop_width, device=rendered_depth.device)
        .unsqueeze(0)
        .expand(crop_height, crop_width)
        .reshape(-1)
    )

    all_rows = rows.unsqueeze(0) + top.unsqueeze(1)
    all_cols = cols.unsqueeze(0) + left.unsqueeze(1)

    extracted_patches_rendered = (
        rendered_depth[:, :, all_rows, all_cols]
        .contiguous()
        .view(
            rendered_depth.shape[0],
            rendered_depth.shape[1],
            num_patches,
            crop_height,
            crop_width,
        )
    )
    extracted_patches_sampled = (
        sampled_depth[:, :, all_rows, all_cols]
        .contiguous()
        .view(
            sampled_depth.shape[0],
            sampled_depth.shape[1],
            num_patches,
            crop_height,
            crop_width,
        )
    )
    extracted_patches_mask = (
        mask[:, :, all_rows, all_cols]
        .contiguous()
        .view(mask.shape[0], mask.shape[1], num_patches, crop_height, crop_width)
        .bool()
    )

    # Remove the second dim (channel dim)
    extracted_patches_rendered = extracted_patches_rendered.squeeze(1)
    extracted_patches_sampled = extracted_patches_sampled.squeeze(1)
    extracted_patches_mask = extracted_patches_mask.squeeze(1)

    # Flatten the tensors
    flattened_sampled_depth = extracted_patches_sampled.view(
        B, num_patches, -1
    )  # B x num_patches x (crop_height * crop_width)
    flattened_rendered_depth = extracted_patches_rendered.view(
        B, num_patches, -1
    )  # B x num_patches x (crop_height * crop_width)
    flattened_mask = extracted_patches_mask.view(
        B, num_patches, -1
    )  # B x num_patches x (crop_height * crop_width)
    flattened_length = flattened_sampled_depth.shape[-1]

    # Argsort sampled depth along the last axis to get sorted indices
    sorted_indices = torch.argsort(flattened_sampled_depth, dim=-1)

    ### Get increasing sequence of indices (idx1 is always less thena idx2)
    outputs = torch.sort(
        torch.randint(
            0,
            flattened_length,
            size=(B, num_patches, num_pairs, 2),
            device=rendered_depth.device,
        ),
        -1,
    )[
        0
    ]  # B x num_patches x num_pairs x 2

    idx1, idx2 = outputs.permute(3, 0, 1, 2)  # 2 x B x num_patches x num_pairs

    ## Flatten sorted_indices
    sorted_indices_batch_flattened = sorted_indices.view(B, -1)
    idx1 = idx1.view(B, -1)
    idx2 = idx2.view(B, -1)

    # Gather the positions from sorted_indices using idx1 and idx2
    positions1 = batched_index_select_nd(sorted_indices_batch_flattened, idx1)
    positions2 = batched_index_select_nd(sorted_indices_batch_flattened, idx2)

    # Use these positions to gather from the flattened depths
    flattended_sampled_depth_batch_flattened = flattened_sampled_depth.view(B, -1)
    flattened_rendered_depth_batch_flattened = flattened_rendered_depth.view(B, -1)
    flattened_mask_batch_flattened = flattened_mask.view(B, -1)
    rendered_depth_values1 = batched_index_select_nd(
        flattened_rendered_depth_batch_flattened, positions1
    )
    rendered_depth_values2 = batched_index_select_nd(
        flattened_rendered_depth_batch_flattened, positions2
    )

    mask_values1 = batched_index_select_nd(flattened_mask_batch_flattened, positions1)
    mask_values2 = batched_index_select_nd(flattened_mask_batch_flattened, positions2)

    mask_combined_values = (mask_values1 & mask_values2).float()

    diff = rendered_depth_values1 - rendered_depth_values2 + margin
    diff = diff * mask_combined_values

    depth_loss = torch.mean(torch.maximum(diff, torch.zeros_like(diff)))

    return depth_loss


def depth_ranking_loss_multi_patch(
    rendered_depth,
    sampled_depth,
    num_patches=16,
    num_pairs=1024,
    margin=1e-4,
    crop_size=(64, 64),
):
    """
    Compute the depth ranking loss between the rendered depth and the sampled depth. The loss is computed by sampling patches from the depth maps and computing the ranking loss between points in the patches.

    :param rendered_depth: The rendered depth map [B 1 H W]
    :param sampled_depth: The sampled depth map [B 1 H W]
    :param num_patches: The number of patches to sample

    :return depth_loss: The depth ranking loss
    """

    B = rendered_depth.shape[0]

    crop_height, crop_width = crop_size
    top = torch.randint(
        0,
        rendered_depth.shape[-2] - crop_height + 1,
        (num_patches,),
        device=rendered_depth.device,
    )
    left = torch.randint(
        0,
        rendered_depth.shape[-1] - crop_width + 1,
        (num_patches,),
        device=rendered_depth.device,
    )

    # Perform the crop - from a B C H W obain B C num_patches crop_height crop_width
    rows = (
        torch.arange(0, crop_height, device=rendered_depth.device)
        .unsqueeze(1)
        .expand(crop_height, crop_width)
        .reshape(-1)
    )
    cols = (
        torch.arange(0, crop_width, device=rendered_depth.device)
        .unsqueeze(0)
        .expand(crop_height, crop_width)
        .reshape(-1)
    )

    all_rows = rows.unsqueeze(0) + top.unsqueeze(1)
    all_cols = cols.unsqueeze(0) + left.unsqueeze(1)

    extracted_patches_rendered = (
        rendered_depth[:, :, all_rows, all_cols]
        .contiguous()
        .view(
            rendered_depth.shape[0],
            rendered_depth.shape[1],
            num_patches,
            crop_height,
            crop_width,
        )
    )
    extracted_patches_sampled = (
        sampled_depth[:, :, all_rows, all_cols]
        .contiguous()
        .view(
            sampled_depth.shape[0],
            sampled_depth.shape[1],
            num_patches,
            crop_height,
            crop_width,
        )
    )

    # Remove the second dim (channel dim)
    extracted_patches_rendered = extracted_patches_rendered.squeeze(1)
    extracted_patches_sampled = extracted_patches_sampled.squeeze(1)

    # Flatten the tensors
    flattened_sampled_depth = extracted_patches_sampled.view(
        B, num_patches, -1
    )  # B x num_patches x (crop_height * crop_width)
    flattened_rendered_depth = extracted_patches_rendered.view(
        B, num_patches, -1
    )  # B x num_patches x (crop_height * crop_width)
    flattened_length = flattened_sampled_depth.shape[-1]

    # Argsort sampled depth along the last axis to get sorted indices
    sorted_indices = torch.argsort(flattened_sampled_depth, dim=-1)

    ### Get increasing sequence of indices (idx1 is always less thena idx2)
    outputs = torch.sort(
        torch.randint(
            0,
            flattened_length,
            size=(B, num_patches, num_pairs, 2),
            device=rendered_depth.device,
        ),
        -1,
    )[
        0
    ]  # B x num_patches x num_pairs x 2

    idx1, idx2 = outputs.permute(3, 0, 1, 2)  # 2 x B x num_patches x num_pairs

    ## Flatten sorted_indices
    sorted_indices_batch_flattened = sorted_indices.view(B, -1)
    idx1 = idx1.view(B, -1)
    idx2 = idx2.view(B, -1)

    # Gather the positions from sorted_indices using idx1 and idx2
    positions1 = batched_index_select_nd(sorted_indices_batch_flattened, idx1)
    positions2 = batched_index_select_nd(sorted_indices_batch_flattened, idx2)

    # Use these positions to gather from the flattened depths
    flattended_sampled_depth_batch_flattened = flattened_sampled_depth.view(B, -1)
    flattened_rendered_depth_batch_flattened = flattened_rendered_depth.view(B, -1)
    rendered_depth_values1 = batched_index_select_nd(
        flattened_rendered_depth_batch_flattened, positions1
    )
    rendered_depth_values2 = batched_index_select_nd(
        flattened_rendered_depth_batch_flattened, positions2
    )

    diff = rendered_depth_values1 - rendered_depth_values2 + margin

    depth_loss = torch.mean(torch.maximum(diff, torch.zeros_like(diff)))

    return depth_loss


def depth_ranking_loss(
    rendered_depth, sampled_depth, num_pairs=1024, margin=1e-4, crop_size=(64, 64)
):
    # Uniformly sample the top-left corner of the crop box
    crop_height, crop_width = crop_size
    top = torch.randint(
        0,
        rendered_depth.shape[-2] - crop_height + 1,
        (1,),
        device=rendered_depth.device,
    )
    left = torch.randint(
        0, rendered_depth.shape[-1] - crop_width + 1, (1,), device=rendered_depth.device
    )

    # Perform the crop
    rendered_depth = rendered_depth[
        :, :, top : top + crop_height, left : left + crop_width
    ].contiguous()
    sampled_depth = sampled_depth[
        :, :, top : top + crop_height, left : left + crop_width
    ].contiguous()

    B, C, H, W = sampled_depth.size()  # Assuming rendered_depth has the same size
    assert (
        C == 1
    ), "Depth map should have one channel! Changing this will break the logic which flattens all channels."

    # Flatten the tensors
    flattened_sampled_depth = sampled_depth.view(B, -1)
    flattened_rendered_depth = rendered_depth.view(B, -1)
    flattened_length = flattened_sampled_depth.shape[-1]

    # Argsort sampled depth along the last axis to get sorted indices
    sorted_indices = torch.argsort(flattened_sampled_depth, dim=-1)

    ### Get increasing sequence of indices (idx1 is always less thena idx2)
    idx1, idx2 = torch.sort(
        torch.randint(
            0, flattened_length, size=(B, num_pairs, 2), device=rendered_depth.device
        ),
        -1,
    )[0].permute(
        2, 0, 1
    )  # B x num_pairs x 2 -> 2 x B x num_pairs

    # Gather the positions from sorted_indices using idx1 and idx2
    positions1 = batched_index_select_nd(sorted_indices, idx1)
    positions2 = batched_index_select_nd(sorted_indices, idx2)

    # Use these positions to gather from the flattened depths
    # sampled_depth_values1 = batched_index_select_nd(flattened_sampled_depth, positions1)
    # sampled_depth_values2 = batched_index_select_nd(flattened_sampled_depth, positions2)
    rendered_depth_values1 = batched_index_select_nd(
        flattened_rendered_depth, positions1
    )
    rendered_depth_values2 = batched_index_select_nd(
        flattened_rendered_depth, positions2
    )

    diff = rendered_depth_values1 - rendered_depth_values2 + margin
    depth_loss = torch.mean(torch.maximum(diff, torch.zeros_like(diff)))

    return depth_loss


def patch_pearson_loss(
    depth_src,  # B 1 H W
    depth_target,  # B 1 H W
    box_p,  # int - patch size
    p_corr,  # float - percentage of patches to use
):
    """

    Compute the patch wise pearson correlation coefficient between the rendered depth and the ground truth depth. Returns 1 - pearson correlation coefficient, so that the loss is minimized when the correlation is maximized.

    As proposed in SparseGS (code taken from paper) - https://arxiv.org/abs/2312.00206

    The paper recommends a patch size of 128 and a p_corr of 0.5

    """

    batch_size = depth_src.shape[0]

    assert (
        batch_size == 1
    ), "Batch size must be 1 for patch pearson loss (can be modified to support batch size > 1)"

    depth_src = depth_src.squeeze()
    depth_target = depth_target.squeeze()

    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    num_box_h = math.floor(depth_src.shape[0] / box_p)
    num_box_w = math.floor(depth_src.shape[1] / box_p)
    max_h = depth_src.shape[0] - box_p
    max_w = depth_src.shape[1] - box_p
    _loss = torch.tensor(0.0, device="cuda")

    # Select the number of boxes based on hyperparameter p_corr
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
    y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0, device="cuda")

    for i in range(len(x_0)):
        pearson_depth_loss = 1 - pearson_corrcoef(
            depth_src[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
            depth_target[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
        )

        _loss += pearson_depth_loss

    return _loss / n_corr


def depth_pearson_loss(
    src: torch.Tensor,  # B C H W
    tgt: torch.Tensor,  # B C H W - the ground truth
):
    """
    Compute the pearson correlation coefficient between the rendered depth and the ground truth depth. Returns 1 - pearson correlation coefficient, so that the loss is minimized when the correlation is maximized.
    """

    batch_size = src.shape[0]

    rendered_depth = src.reshape(batch_size, -1)
    pred_depth = tgt.reshape(batch_size, -1)

    pearson_corr_loss = 0
    for idx in range(batch_size):
        pearson_corr_loss += 1.0 - pearson_corrcoef(
            rendered_depth[idx], pred_depth[idx]
        )

    return pearson_corr_loss


def depth_ranking_smooth_loss(
    src: torch.Tensor,  # B C H W
    tgt: torch.Tensor,  # B C H W - the ground truth
):
    """
    Compute the ranking loss between the rendered depth and the ground truth depth. Returns the sum of the ranking loss in the horizontal and vertical directions.

    Ref: Sparse NeRF
    """

    x = src
    y = tgt

    x_diff_vertical = x[:, :, 1:, :] - x[:, :, :-1, :]
    y_diff_vertical = y[:, :, 1:, :] - y[:, :, :-1, :]

    x_diff_horizontal = x[:, :, :, 1:] - x[:, :, :, :-1]
    y_diff_horizontal = y[:, :, :, 1:] - y[:, :, :, :-1]

    differing_signs_vertical = torch.sign(x_diff_vertical) != torch.sign(
        y_diff_vertical
    )
    differing_signs_horizontal = torch.sign(x_diff_horizontal) != torch.sign(
        y_diff_horizontal
    )

    horizontal_loss = torch.nanmean(
        x_diff_horizontal[differing_signs_horizontal]
        * torch.sign(x_diff_horizontal[differing_signs_horizontal])
    )
    vertical_loss = torch.nanmean(
        x_diff_vertical[differing_signs_vertical]
        * torch.sign(x_diff_vertical[differing_signs_vertical])
    )

    ranking_loss = horizontal_loss + vertical_loss

    return ranking_loss


# Alex
def get_scale_translation(
    x: np.ndarray,  # (N,)
    y: np.ndarray,  # (N,)
):
    """
    Given two N dimensions tensors, compute a solution to a * x + b = y. Return a, b
    """

    x = x[:, None]  # (N, 1)
    x = np.concatenate([x, np.ones(shape=x.shape)], -1)  # (N, 2)
    psuedo = np.linalg.inv(x.T @ x) @ x.T

    scale, translation = (psuedo @ y[..., None]).squeeze()

    return scale, translation


def align_depths(
    source_depth: torch.Tensor,  # B 1 H W
    target_depth: torch.Tensor,  # B 1 H W
    mask: torch.Tensor = None,  # B 1 H W
):
    """
    Given two depth maps, align one with the other. If a mask is provided, will use only the values within the mask
    """

    batch_size, _, _, _ = source_depth.shape

    output_depth = source_depth.clone()

    for i in range(batch_size):

        source_depth_i = source_depth[i].squeeze()  # H W
        target_depth_i = target_depth[i].squeeze()

        if mask is not None:
            mask_i = mask[i].squeeze()
            source_depth_i = source_depth_i[mask_i]
            target_depth_i = target_depth_i[mask_i]

        scale, translation = get_scale_translation(
            source_depth_i.flatten().cpu().detach().numpy(),
            target_depth_i.flatten().cpu().detach().numpy(),
        )

        output_depth[i] = translation + scale * source_depth

    return output_depth


def blend_depths(
    source_depth: torch.Tensor,  # B 1 H W
    target_depth: torch.Tensor,  # B 1 H W
    mask_new: torch.Tensor = None,  # B 1 H W
):
    """
    Given two depth maps and a mask, ensures that the content of the source depth map within the mask blends smoothly with the contents of target within the inverse mask.

    source * mask ~= target_depth * (1 - mask)

    """

    batch_size = source_depth.shape[0]
    mask_new = mask_new.float()

    mask_new_dilated = mask_new.clone()
    # for i in range(10):
    #     mask_new_dilated = kornia.morphology.dilation(mask_new_dilated, torch.ones(5, 5).to(mask_new.device))

    combined_depth = source_depth.clone()

    for b_i in range(batch_size):  # for all batch elements

        combined_depth[b_i] = source_depth[b_i] * mask_new[b_i] + target_depth[b_i] * (
            1 - mask_new[b_i]
        )  # Use predicted source depth within the mask and GT rendered depth outside mask

        import matplotlib.pyplot as plt

        plt.imshow(combined_depth[b_i].squeeze().cpu().numpy())
        plt.colorbar()
        plt.title("Combined depth")
        plt.show()

        source_depth_i = source_depth[b_i]
        target_depth_i = target_depth[b_i]
        mask_new_i = mask_new[b_i]
        mask_new_dilated_i = mask_new_dilated[b_i]

        for i in range(10000):
            depth = apply_depth_smoothing(
                combined_depth[b_i].float().squeeze(), mask_new_i.squeeze()
            ).unsqueeze(0)
            combined_depth[b_i] = depth * mask_new_i + target_depth_i * (1 - mask_new_i)
            # break

    return combined_depth


def apply_depth_smoothing(image: torch.Tensor, mask: torch.Tensor):  # H W  # H W
    """
    Smooth an image along the edges of a mask

    From text2room
    """

    def dilate(x, k=3):
        x = torch.nn.functional.conv2d(
            x.float()[None, None, ...],
            torch.ones(1, 1, k, k).to(x.device),
            padding="same",
        )
        return x.squeeze() > 0

    def sobel(x):
        flipped_sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).to(x.device)
        flipped_sobel_x = torch.stack([flipped_sobel_x, flipped_sobel_x.t()]).unsqueeze(
            1
        )

        x_pad = torch.nn.functional.pad(
            x.float()[None, None, ...], (1, 1, 1, 1), mode="replicate"
        )

        x = torch.nn.functional.conv2d(x_pad, flipped_sobel_x, padding="valid")
        dx, dy = x.unbind(dim=-3)
        return torch.sqrt(dx**2 + dy**2).squeeze()
        # new content is created mostly in x direction, sharp edges in y direction are wanted (e.g. table --> wall)
        # return dx.squeeze()

    edges = sobel(mask)
    dilated_edges = dilate(edges, k=21)

    # import matplotlib.pyplot as plt
    # plt.imshow(dilated_edges.squeeze().cpu().numpy())
    # plt.show()

    # image_copy = image.clone()
    # image_copy[dilated_edges] = 0
    # plt.imshow(image_copy.squeeze().cpu().numpy())
    # plt.show()

    img_numpy = image.float().cpu().numpy()

    blur_gaussian = cv2.blur(img_numpy, (5, 5), 0)
    blur_gaussian = torch.from_numpy(blur_gaussian).to(image)

    # plt.imshow(blur_gaussian.squeeze().cpu().numpy())
    # plt.title("Gaussian blur")
    # plt.show()

    # print(edges.shape, edges.dtype)
    # image_smooth = torch.where(edges[None, None, ...].bool(), blur_gaussian, image)
    image_smooth = torch.where(dilated_edges.bool(), blur_gaussian, image)
    # image_smooth = blur_gaussian
    return image_smooth


def compute_laplacian(x):

    import torch.nn.functional as F

    # Laplacian operator
    laplacian_kernel = (
        torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        .view(1, 1, 3, 3)
        .cuda()
    )

    # Compute Laplacian of target image
    laplacian = F.conv2d(x, laplacian_kernel, padding=1)

    del laplacian_kernel

    return laplacian


def compute_gt_gradient(pred_depth, rendered_depth, mask):

    import torch.nn.functional as F

    # Laplacian operator
    laplacian_kernel = (
        torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        .view(1, 1, 3, 3)
        .cuda()
    )

    # Compute Laplacian of rendered depth
    rendered_laplacian = F.conv2d(rendered_depth, laplacian_kernel, padding=1)

    # Compute Laplacian of predicted depth
    pred_laplacian = F.conv2d(pred_depth, laplacian_kernel, padding=1)

    # Compute blended Laplacian
    # blended_laplacian = mask * rendered_laplacian + (1 - mask) * pred_laplacian
    blended_laplacian = pred_laplacian

    del laplacian_kernel

    return blended_laplacian


# Parts taken from https://github.com/owenzlz/DeepImageBlending/
def poisson_blend(depth_pred, depth_render, mask):
    """

    Use poisson blending to blend the predicted depth with the rendered depth. The blending is done in the outpainted region of the mask.

    :param depth_pred: The predicted depth map [B 1 H W]
    :param depth_render: The rendered depth map [B 1 H W]
    :param mask: The outpainting mask [B 1 H W]

    """

    assert (
        depth_pred.shape == depth_render.shape
    ), "Depth pred and depth render must have the same shape"
    assert depth_pred.shape[0] == 1, "Batch size must be 1 for poisson blending"

    depth_pred = depth_pred[0]
    depth_render = depth_render[0]
    mask = mask[0]

    import torch.optim as optim

    num_steps = 500

    # Initialise the depth to optimise
    mask_rendered = (depth_render > 0).bool().to(depth_render.device)
    optim_depth = depth_pred.clone()
    optim_depth[mask_rendered] = depth_render[mask_rendered]

    # Init Optimizer
    optimizer = optim.LBFGS([optim_depth.requires_grad_()])
    mse = torch.nn.MSELoss()

    # Init the gradient Ground truth
    gt_gradient = compute_gt_gradient(depth_pred, depth_render, mask_rendered)

    # Dilate teh outpainting mask
    mask_outpaint_dilated = (
        kornia.morphology.dilation(
            mask.unsqueeze(1), kernel=torch.ones((5, 5), device=depth_pred.device)
        )
        .squeeze(1)
        .bool()
    )

    # Compute hte intersection mask
    mask = mask_rendered.bool() & mask_outpaint_dilated

    # Loss
    losses = [10]

    # Plot all the masks

    for i in range(num_steps):

        def closure():

            # We want L2 Loss on the gradient in the outpainted region
            # Compute Laplacian Gradient of Blended Image
            pred_gradient = compute_laplacian(optim_depth)

            # Compute Gradient Loss
            grad_loss = 0
            grad_loss += mse(
                pred_gradient[mask_outpaint_dilated], gt_gradient[mask_outpaint_dilated]
            )
            grad_loss /= len(pred_gradient)

            # The depth rendered must be the same
            grad_loss += mse(optim_depth[mask], depth_render[mask])

            grad_loss *= 1e4

            loss = grad_loss
            optimizer.zero_grad()
            loss.backward()

            return loss

        loss = optimizer.step(closure)

        # if np.abs(loss.item() - losses[-1]) < 1e-7:
        #     break

        losses.append(loss.item())

    # Clear up memory
    del gt_gradient
    del mask_rendered
    del mask_outpaint_dilated
    del mask

    # Delete optimizer
    del optimizer

    # if self.cfg.display:
    #     plt.imshow(optim_depth[0].detach().cpu().numpy())
    #     plt.title("optim_depth - after")
    #     plt.colorbar()
    #     plt.show()

    return optim_depth.detach()


if __name__ == "__main__":

    depth = torch.rand(1, 1, 512, 512)
    pred_depth = torch.rand(1, 1, 512, 512)

    loss = depth_ranking_loss_multi_patch(depth, pred_depth)
