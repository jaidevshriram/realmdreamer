import numpy as np
import torch
from functools import reduce


def sparse_bilateral_filtering(
    depth,
    image,
    config,
    HR=False,
    mask=None,
    gsHR=True,
    edge_id=None,
    num_iter=None,
    num_gs_iter=None,
    spdb=False,
):
    """
    config:
    - filter_size
    """
    import time

    save_images = []
    save_depths = []
    save_discontinuities = []
    vis_depth = depth.copy()
    backup_vis_depth = vis_depth.copy()

    depth_max = vis_depth.max()
    depth_min = vis_depth.min()
    vis_image = image.copy()
    for i in range(num_iter):
        if isinstance(config["filter_size"], list):
            window_size = config["filter_size"][i]
        else:
            window_size = config["filter_size"]
        vis_image = image.copy()
        save_images.append(vis_image)
        save_depths.append(vis_depth)
        u_over, b_over, l_over, r_over = vis_depth_discontinuity(
            vis_depth, config, mask=mask
        )
        vis_image[u_over > 0] = np.array([0, 0, 0])
        vis_image[b_over > 0] = np.array([0, 0, 0])
        vis_image[l_over > 0] = np.array([0, 0, 0])
        vis_image[r_over > 0] = np.array([0, 0, 0])

        discontinuity_map = (u_over + b_over + l_over + r_over).clip(0.0, 1.0)
        discontinuity_map[depth == 0] = 1
        save_discontinuities.append(discontinuity_map)
        if mask is not None:
            discontinuity_map[mask == 0] = 0
        vis_depth = bilateral_filter_logic(
            vis_depth,
            config,
            discontinuity_map=discontinuity_map,
            HR=HR,
            mask=mask,
            window_size=window_size,
        )

    return save_images, save_depths


def vis_depth_discontinuity(depth, config, vis_diff=False, label=False, mask=None):
    """
    config:
    -
    """
    if label == False:
        disp = 1.0 / depth
        u_diff = (disp[1:, :] - disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] - disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] - disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] - disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > config["depth_threshold"]).astype(np.float32)
        b_over = (np.abs(b_diff) > config["depth_threshold"]).astype(np.float32)
        l_over = (np.abs(l_diff) > config["depth_threshold"]).astype(np.float32)
        r_over = (np.abs(r_diff) > config["depth_threshold"]).astype(np.float32)
    else:
        disp = depth
        u_diff = (disp[1:, :] * disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] * disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] * disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] * disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > 0).astype(np.float32)
        b_over = (np.abs(b_diff) > 0).astype(np.float32)
        l_over = (np.abs(l_diff) > 0).astype(np.float32)
        r_over = (np.abs(r_diff) > 0).astype(np.float32)
    u_over = np.pad(u_over, 1, mode="constant")
    b_over = np.pad(b_over, 1, mode="constant")
    l_over = np.pad(l_over, 1, mode="constant")
    r_over = np.pad(r_over, 1, mode="constant")
    u_diff = np.pad(u_diff, 1, mode="constant")
    b_diff = np.pad(b_diff, 1, mode="constant")
    l_diff = np.pad(l_diff, 1, mode="constant")
    r_diff = np.pad(r_diff, 1, mode="constant")

    if vis_diff:
        return [u_over, b_over, l_over, r_over], [u_diff, b_diff, l_diff, r_diff]
    else:
        return [u_over, b_over, l_over, r_over]


def bilateral_filter_logic(
    depth, config, discontinuity_map=None, HR=False, mask=None, window_size=False
):
    sort_time = 0
    replace_time = 0
    filter_time = 0
    init_time = 0
    filtering_time = 0
    sigma_s = config["sigma_s"]
    sigma_r = config["sigma_r"]
    if window_size == False:
        window_size = config["filter_size"]
    midpt = window_size // 2
    ax = np.arange(-midpt, midpt + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    if discontinuity_map is not None:
        spatial_term = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_s**2))

    # padding
    depth = depth[1:-1, 1:-1]
    depth = np.pad(depth, ((1, 1), (1, 1)), "edge")
    pad_depth = np.pad(depth, (midpt, midpt), "edge")
    if discontinuity_map is not None:
        discontinuity_map = discontinuity_map[1:-1, 1:-1]
        discontinuity_map = np.pad(discontinuity_map, ((1, 1), (1, 1)), "edge")
        pad_discontinuity_map = np.pad(discontinuity_map, (midpt, midpt), "edge")
        pad_discontinuity_hole = 1 - pad_discontinuity_map
    # filtering
    output = depth.copy()
    pad_depth_patches = rolling_window(pad_depth, [window_size, window_size], [1, 1])
    if discontinuity_map is not None:
        pad_discontinuity_patches = rolling_window(
            pad_discontinuity_map, [window_size, window_size], [1, 1]
        )
        pad_discontinuity_hole_patches = rolling_window(
            pad_discontinuity_hole, [window_size, window_size], [1, 1]
        )

    if mask is not None:
        pad_mask = np.pad(mask, (midpt, midpt), "constant")
        pad_mask_patches = rolling_window(pad_mask, [window_size, window_size], [1, 1])
    from itertools import product

    if discontinuity_map is not None:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if mask is not None and mask[pi, pj] == 0:
                    continue
                if discontinuity_map is not None:
                    if bool(pad_discontinuity_patches[pi, pj].any()) is False:
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = pad_discontinuity_hole_patches[pi, pj]
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size // 2, window_size // 2]
                if discontinuity_map is not None:
                    coef = discontinuity_holes.astype(np.float32)
                    if mask is not None:
                        coef = coef * pad_mask_patches[pi, pj]
                else:
                    range_term = np.exp(
                        -((depth_patch - patch_midpt) ** 2) / (2.0 * sigma_r**2)
                    )
                    coef = spatial_term * range_term
                if coef.max() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.max() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef / (coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]
    else:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if discontinuity_map is not None:
                    if (
                        pad_discontinuity_patches[pi, pj][
                            window_size // 2, window_size // 2
                        ]
                        == 1
                    ):
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = 1.0 - discontinuity_patch
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size // 2, window_size // 2]
                range_term = np.exp(
                    -((depth_patch - patch_midpt) ** 2) / (2.0 * sigma_r**2)
                )
                if discontinuity_map is not None:
                    coef = spatial_term * range_term * discontinuity_holes
                else:
                    coef = spatial_term * range_term
                if coef.sum() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.sum() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef / (coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]

    return output


def rolling_window(a, window, strides):
    assert (
        len(a.shape) == len(window) == len(strides)
    ), "'a', 'window', 'strides' dimension mismatch"
    shape_fn = lambda i, w, s: (a.shape[i] - w) // s + 1
    shape = [shape_fn(i, w, s) for i, (w, s) in enumerate(zip(window, strides))] + list(
        window
    )

    def acc_shape(i):
        if i + 1 >= len(a.shape):
            return 1
        else:
            return reduce(lambda x, y: x * y, a.shape[i + 1 :])

    _strides = [acc_shape(i) * s * a.itemsize for i, s in enumerate(strides)] + list(
        a.strides
    )

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)


def bilateral_filter(
    img: torch.Tensor, depth: torch.Tensor, depth_threshold=0.01  # B C H W  # B 1 H W
):

    # Config for 3D photography
    config = {}
    config["gpu_ids"] = 0
    config["extrapolation_thickness"] = 60
    config["extrapolate_border"] = True
    config["depth_threshold"] = depth_threshold
    config["redundant_number"] = 12
    config["ext_edge_threshold"] = 0.002
    config["background_thickness"] = 70
    config["context_thickness"] = 140
    config["background_thickness_2"] = 70
    config["context_thickness_2"] = 70
    config["log_depth"] = True
    config["depth_edge_dilate"] = 10
    config["depth_edge_dilate_2"] = 5
    config["largest_size"] = 512
    config["repeat_inpaint_edge"] = True
    config["ply_fmt"] = "bin"

    config["save_ply"] = False
    config["save_obj"] = False

    config["sparse_iter"] = 10
    config["filter_size"] = [7, 7, 5, 5, 5, 5, 5, 5, 5, 5]
    config["sigma_s"] = 4.0
    config["sigma_r"] = 0.5

    B = img.shape[0]
    output = torch.zeros_like(depth)

    for i in range(B):

        # Convert to numpy
        img_np = img[i].detach().cpu().permute(1, 2, 0).numpy()
        depth_np = depth[i].detach().cpu().numpy().squeeze()

        # Run the filter
        vis_images, vis_depths = sparse_bilateral_filtering(
            depth_np,
            img_np * 255,
            config,
            num_iter=config["sparse_iter"],
            spdb=False,
            HR=True,
        )

        # Convert back to torch
        vis_depths = (
            torch.from_numpy(vis_depths[-1]).to(img.device).float().to(depth.device)
        )

        # Plot the output
        # plt.imshow(vis_images[-1])
        # plt.title("vis_image")
        # plt.colorbar()
        # plt.show()

        # Save the output
        output[i] = vis_depths

    return output
